from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# This evaluates the InceptionV3 model on the test dataset.
# Run 'sbatch MyJob3.slurm' to run it. 

model = load_model('inception_v3_plants.h5')
model.summary()


plant_color = 'color' # 3 categories: color/grayscale/segmented
test_dir = 'plant_data_small_3/'+plant_color+'/test'
test_datagen = ImageDataGenerator(rescale=1./255) # Don't augment test data!
test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(224,224), batch_size=128, class_mode='categorical')


# Evaluate model on testing data
print('Testing model on test data...')
test_loss, test_acc = model.evaluate(x=test_generator, verbose=1) # Add 'steps' parameter? 
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)