import matplotlib.pyplot as plt
import pytesseract
import cv2


def open_image(img_path):
    carplate_img = cv2.imread(img_path)
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(carplate_img)
    plt.show()
    return carplate_img


def carplate_extract(image, carplate_hear_cascade):
    carplate_rects = carplate_hear_cascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img


def enlarge_img(image, scale_precent):
    height = int(image.shape[0] * scale_precent / 100)
    width = int(image.shape[1] * scale_precent / 100)
    dim = (width, height)
    plt.axis('off')
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def main():
    carplate_img_rgb = open_image(
        img_path='/Users/floyz/VSCode/Python/project-mars/model/data/car2.jpeg')
    carplate_hear_cascade = cv2.CascadeClassifier(
        '/Users/floyz/VSCode/Python/project-mars/model/hear/hear_cascade_russin_plate_number.xml')

    carplate_extract_img = carplate_extract(
        carplate_img_rgb, carplate_hear_cascade)
    carplate_extract_img = enlarge_img(carplate_extract_img, scale_precent=150)
    plt.imshow(carplate_extract_img)
    plt.show()

    carplate_extract_img_gray = cv2.cvtColor(
        carplate_extract_img, cv2.COLOR_RGB2GRAY)
    plt.axis('off')
    plt.imshow(carplate_extract_img_gray, cmap='gray')
    plt.show()

    # pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

    print('Номер автомобиля: ', pytesseract.image_to_string(
        carplate_extract_img_gray,
        config='--psm 6 --oem 3 -c "tessedit_char_whitelist=ABCDEFGHIJKLMNOPORSTUVWXYZ0123456789"')
    )


if __name__ == '__main__':
    main()
