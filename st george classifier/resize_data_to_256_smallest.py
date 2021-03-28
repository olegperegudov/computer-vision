import cv2
import albumentations as A
import os

path = r'' 
read_path = os.path.join(path, 'test')
write_path = os.path.join(path, 'test_resized')

images = os.listdir(read_path)

for idx, image in enumerate(images):

    if images[idx][-3:] == 'jpg':

        image = cv2.imread(os.path.join(read_path, image))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        transform = A.Compose([
            A.SmallestMaxSize(256)
        ])

        transformed = transform(image=image)
        transformed_image = transformed["image"]

        cv2.imwrite(os.path.join(write_path, images[idx]), transformed_image)

    if (idx+1) % 250 == 0:
        print(f'{idx+1} images are done.')

print('All done.')
