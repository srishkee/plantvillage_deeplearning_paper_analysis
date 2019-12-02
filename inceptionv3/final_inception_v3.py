import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import pandas as pd

# This script builds the InceptionV3 model.
# Run 'sbatch MyJob2.slurm' to run it. 

def print_training_data(acc, val_acc, loss, val_loss):
	print('Training Accuracy:\n', acc) 
	print('Validation Accuracy:\n', val_acc) 
	print('Training Loss:\n', loss) 
	print('Validation Loss:\n', val_loss) 

def write_data_to_file(acc, val_acc, loss, val_loss):
	# Write data to .csv file
	# Note: Clear data.csv each time! After clearing, add '0' to make it non-empty
	open('data.csv', 'w+').close() # Clear file before writing to it (and create if nonexistent)
	with open('data.csv', 'w') as f:
		f.write('0') # Add a value
	f.close()
	print('Writing data to .csv file...')
	data = pd.read_csv('data.csv', 'w') 
	data.insert(0,"Training Acc", acc)
	data.insert(1,"Training Loss", val_acc)
	data.insert(2,"Validation Acc", loss)
	data.insert(3,"Validation Loss", val_loss)
	data.to_csv('data.csv')
	print('Finished writing data!')

def plot_graphs(acc, val_acc, loss, val_loss):
	# Plot results
	print("Starting plotting...") 
	epochs = range(1, len(acc)+1)
	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and Validation accuracy')
	plt.legend()
	plt.savefig('Plants_TrainingAcc.png')
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.legend()
	plt.ylim([0,4]) # Loss should not increase beyond 4! 
	plt.savefig('Plants_TrainingLoss.png')
	plt.figure()
	plt.show()
	print('Finished plotting!')



# Instantiate a Inception V3 model
print ("Initializing model...")
# Fully connected layer: each neuron receives input from each previous layer 
incep_v3 = InceptionV3(include_top=False, weights=None, input_shape=(224,224,3), pooling='avg')
print ("incep_v3 output: ", incep_v3.output) 

x = incep_v3.output
x = Dropout(0.5)(x) # Add a dropout layer				
x = Dense(1024, activation='relu')(x) # Add a fully-connected layer 
x = Dropout(0.5)(x) # Add a dropout layer				
predictions = Dense(38, activation='softmax')(x) 
print("predictions output: ", predictions.shape)

model = Model(inputs=incep_v3.input, outputs=predictions)
model.summary()
print ("model output: ", model.output)

# Loss is the error to be minimized. Cross-entropy = -log(error)
print ("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['acc']) 

# Only need local folder path! (Not absolute)
plant_color = 'color' # 3 categories: color/grayscale/segmented
train_dir = 'plant_data_small_3/'+plant_color+'/train'
validation_dir = 'plant_data_small_3/'+plant_color+'/validation'
test_dir = 'plant_data_small_3/'+plant_color+'/test'

# Generate batches of tensor image data + rescales all images by 255 (scales down)
# train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator( # Augment training data 
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.1,
	height_shift_range=0.1,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255) # Don't augment validation data!
test_datagen = ImageDataGenerator(rescale=1./255) # Don't augment test data!

# Takes path to a directory and generates batches of augmented data 
train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(224,224), batch_size=128, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(224,224), batch_size=128, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(224,224), batch_size=128, class_mode='categorical')

print ("Fitting model to data...")
history = model.fit_generator(train_generator, steps_per_epoch=50, epochs=50, validation_data=validation_generator, validation_steps=40) 
print("Fit model to data successfully!") 


# Save training parameters
print('Obtaining training data...')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Save model 
print ("Saving model...")
model.save('inception_v3_plants.h5')

print_training_data(acc, val_acc, loss, val_loss)
write_data_to_file(acc, val_acc, loss, val_loss)
plot_graphs(acc, val_acc, loss, val_loss)


# Evaluate model on testing data
print('Testing model on test data...')
test_loss, test_acc = model.evaluate(x=test_generator, verbose=1) # Add 'steps' parameter? 
print('test_loss: ', test_loss)
print('test_acc: ', test_acc)

print('Finished all!')

