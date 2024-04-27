#!/usr/bin/python3

import sys
import cv2
import torch
import argparse
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

def main(args):
    data_dir = Path(args.data).resolve()
    unaligned_dir = data_dir.joinpath('photos_raw')
    aligned_dir = data_dir.joinpath('photos_aligned_faces')

    showFaces = args.show_faces
    saveFaces = args.save_faces
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on ', device)
    mtcnn = MTCNN(keep_all=False, select_largest=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    faces = []
    for dir_path in unaligned_dir.iterdir():
        person_name = dir_path.name
        print('\033[92m'+person_name+'\033[0m')

        output_path_root = aligned_dir.joinpath(person_name)
        for file_path in dir_path.iterdir():
            output_file_path = output_path_root.joinpath(file_path.name)
            print(file_path.name)
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgCropped = mtcnn(img, save_path = str(output_file_path) if saveFaces else None)
            if imgCropped is None:
                print('No face found: ' + file_path.name)
            else:
                if showFaces:
                    cvImg = imgCropped.permute(1, 2, 0).numpy()
                    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)
                    cv2.imshow('image', cvImg)
                    cv2.waitKey()

                torchImg = imgCropped.unsqueeze(0).to(device)
                embedding = resnet(torchImg).detach().cpu()
                faces.append([person_name, file_path.name, embedding])

    torch.save(faces, str(data_dir.joinpath('embeddings.pt')))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default='../data',
                        help='Path to the data folder (which contains the photos_raw folder)')

    parser.add_argument('--no_save_faces', default=True, dest='save_faces', action='store_false',
                        help='Specify that cropped faces should not be saved to photos_aligned_faces folder for review')

    parser.add_argument('--show_faces', default=False, dest='show_faces', action='store_true',
                        help='Specify that cropped faces should be displayed on-screen as they are extracted')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))