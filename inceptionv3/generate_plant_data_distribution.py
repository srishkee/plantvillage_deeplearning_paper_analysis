import os, shutil

# This file generates the 60-20-20 train/validation/test data ratio for the InceptionV3 model 

path = '/scratch/user/skumar55/plantvillage_deeplearning_paper_dataset/raw'
dst_folder = '/scratch/user/skumar55/plant_data_small_3'

for plant_color in os.listdir(path): # Iterate through color/grayscale/segmented

	print(plant_color)

	for (idx,plant_species) in enumerate(os.listdir(path+'/'+plant_color)): # Iterate through all plant species
		
		print(idx, plant_species) # Print plant species & index

		plant_species_sum = 0 # Number of plant imgs in this plant species
		for plant_img in os.listdir(path+'/'+plant_color+'/'+plant_species): # Iterate through all plant images
			plant_species_sum += 1 # Get sum of all plant images in folder

		# Get train-test ratios of plant species
		train_ratio = int(0.6*plant_species_sum)
		validation_ratio = int(0.2*plant_species_sum)
		test_ratio = int(0.2*plant_species_sum)
			
		imgs = []
		imgs = os.listdir(path+'/'+plant_color+'/'+plant_species) # Store all plant species images
		
		for i in range(0, train_ratio): # Move training images
			my_img = imgs[i]
			src = os.path.join(path+'/'+plant_color+'/'+plant_species, my_img)
			dst = os.path.join(dst_folder+'/'+plant_color+'/'+'train'+'/'+str(idx), my_img)
			shutil.copyfile(src, dst)
		print('Copied all train images!')

		for i in range(train_ratio, train_ratio+validation_ratio): # Move validation images
			my_img = imgs[i]
			src = os.path.join(path+'/'+plant_color+'/'+plant_species, my_img)
			dst = os.path.join(dst_folder+'/'+plant_color+'/'+'validation'+'/'+str(idx), my_img)
			shutil.copyfile(src, dst)
		print('Copied all validation images!')

		for i in range(train_ratio+validation_ratio, train_ratio+validation_ratio+test_ratio): # Move testing images
			my_img = imgs[i]
			src = os.path.join(path+'/'+plant_color+'/'+plant_species, my_img)
			dst = os.path.join(dst_folder+'/'+plant_color+'/'+'test'+'/'+str(idx), my_img)
			shutil.copyfile(src, dst)
		print('Copied all test images!')



# '''
# 1. Get sum of images in folder
# 2. Get train/validation/test set ratios (60-20-20)
# 3. Run a for loop through those numbers
# 4. Move all those images into plant_data_small_3/TYPE/SPECIES folder 
# '''