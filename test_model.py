'''
---------------------------------------------------
Devin Suy : https://devinsuy.com/
Dataset: https://www.kaggle.com/c/dogs-vs-cats/data
---------------------------------------------------

Test trained model, predict {cat, dog} on new unseen image
'''
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

BASE_DIR = 'data/predict/'
IMG_NAME = '1.jpg'

# Load and process image into 200x200 pixels (shape model was trained with)
test_image = img_to_array(load_img(BASE_DIR + IMG_NAME, target_size=(200, 200)))
test_image = test_image.reshape(1, 200, 200, 3)
test_image = test_image.astype('float32')
test_image = test_image - [123.68, 116.779, 103.939]

# Load previously trained model
model = load_model('cat_dog_classifer.h5')

# Classes are: {0 -> cat, 1 -> dog}
type = model.predict(test_image)[0]

# Output results
if type == 0:
    print("\n\n------------------------------------------")
    print("CAT")
    print("------------------------------------------")
else:
    print("\n\n------------------------------------------")
    print("DOG")
    print("------------------------------------------")   