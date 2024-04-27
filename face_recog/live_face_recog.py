#!/usr/bin/python3

import rclpy
from rclpy.node import Node

import cv2
import torch
from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image
from vision_interfaces.msg import Detections
from vision_interfaces.msg import Detection

class Face:

    def __init__(self, box, face_prob, img, name, ident_prob):
        self.box = box.astype(int)    #bounding box of face
        self.face_prob = face_prob    #prob of box containing a face
        self.img = img                #the image in framed by box (a face)
        self.name = name              #predicted name for face
        self.ident_prob = ident_prob  #probability of correct name

#---------------------------------------------------------------------------

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            box = face.box.astype(int)
            
            cv2.rectangle(frame,
                          (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 2)
                          
            if face.name is not None:
                cv2.putText(frame, 
                            face.name+':'+str(int(100*face.ident_prob)),
                            (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, 
                            lineType=2)

    cv2.putText(frame, 
                str(len(faces)) + " faces found", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, 
                lineType=2)

#---------------------------------------------------------------------------

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('show_faces', False)
        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.pub_detections = self.create_publisher(Detections, '/faces/detections', 1)

    def broadcast_identifications(self, faces, imgMsg):
        msg = Detections()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = imgMsg.header.frame_id
        msg.image_header = imgMsg.header
           
        for face in faces:
            box = face.box
            detection = Detection();
            detection.visual_class = face.name
            detection.probability = float(face.ident_prob)
            detection.xmin = int(box[0])
            detection.ymin = int(box[1])
            detection.xmax = int(box[2])
            detection.ymax = int(box[3])
            msg.detections.append(detection)

        self.pub_detections.publish(msg)

    def listener_callback(self, imgMsg):
        
        #convert image message to an opencv image
        try:
           cv_image = bridge.imgmsg_to_cv2(imgMsg, "rgb8") #This format is vital, or mtcnn won't work !
        except CvBridgeError as e:
           print(e)
           return
           
        boxes, face_probs = mtcnn.detect(cv_image)
        
        faces = []
        if boxes is not None:

            face_imgs = mtcnn.extract(cv_image, boxes, None)
            embeddings = resnet(face_imgs.to(device))
            yhat = model(embeddings)
            winners = torch.max(yhat, 1)
            ident_probs = winners.values.detach().cpu()
            ids = winners.indices.detach().cpu()
            names = model.label_encoder.inverse_transform(ids)

            for (box, face_prob, face_img, name, ident_prob) in zip(boxes, face_probs, face_imgs, names, ident_probs):
                if ident_prob>0.85: 
                    faces.append(Face(box, face_prob, face_img, name, ident_prob))

        if show_faces:
            cv_image_out = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            add_overlays(cv_image_out, faces)
            cv2.imshow("img", cv_image_out)
            
        self.broadcast_identifications(faces, imgMsg)

def main(args=None):
    global bridge, device,  mtcnn, resnet, model, show_faces
    
    rclpy.init(args=args)

    bridge = CvBridge()
    cv2.startWindowThread()

    #set up mtcnn(face locator) and resnet(face recognizer) on gpu if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on: ', device)
    mtcnn = MTCNN(keep_all=True, select_largest=False, min_face_size= 40, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model = torch.load('/home/paul/Documents/ROS2/ros2_ws/src/face_recog/data/face_classifier.pt')

    minimal_subscriber = MinimalSubscriber()
    show_faces = minimal_subscriber.get_parameter('show_faces').get_parameter_value().bool_value
    
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node() # (explicit destruction is optional - gc will be do it automatically
    rclpy.shutdown()


if __name__ == '__main__':
    main()
