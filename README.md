Final Capstone - Deep Learning - Chest Xrays Comparing Normal Chest Xrays to Viral and Bacterial Pneunomia Chest Xrays.

Brian Farrelly, Thinkful Data Science Immersion Bootcamp

May 21, 2020






---



---



---



---



This Header will be included in all five project files. modeltest.ipynb,
bacterial_pneumonia_xrays.ipynb, bacterial_pneumonia_xrays_robust.ipynb,
viral_penumonia_xrays.ipynb, viral_pneumonia_xrays_robust.ipynb. "Chest Xray" will be replaced the appreviation CXR. Viral Penumonia will be referred to as Viral. And Bacterial Pneumonia will be referred to as Bacterial.




---



---



---



---




1. Wrangle your data. Get it into the notebook in the best form possible for your analysis and model building.

Data pulled from: 
https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

Stored on a mounted google drive with 1300 training files for normal, 1300 viral pneumonia files and 1300 bacterial pneumonia files. 

The data is loaded into a generator and that generator is used to train the model.

The files were easy to classify as normal files were indicated by strictly defined file names that started with "IM" or "NORMAL" for normal. Viral pneumonia files with "virus" in the filename e.g. person80_virus_150.jpeg.
Bacterial pneumonia files with "bacteria" in the filename e.g. person1_bacteria_1.jpeg.
    



2. Explore your data. Make visualizations and conduct statistical analyses to explain what’s happening with your data, why it’s interesting, and what features you intend to take advantage of for your modeling. The number of files are listed by category in the dataset summary that was provided. Also the files
are clearing delineated by a strict filename convention listed above.

corona.head(20)

3. Build a modeling pipeline. Your model should be built in a coherent pipeline of linked stages that is efficient and easy to implement.

The notebooks attached will be easy to run as a sanity test just make sure to change the model.fit() with 16 steps_per_epoch and 8 validation steps. The whole notbook should run start to finish without interruption. Access to my google drive may be an issue. 
I will make sure to make them read only for the public. There should be no need to run Tensorboard or use "model save per epoch"
callbacks but they are there if needed. 

https://drive.google.com/open?id=1W9yqHpRDIDt2mzczRBmsF-lQSHSlyoxn

https://drive.google.com/open?id=16_2qCY8dcjeh8P1JZsNUluvGyi0TFvvZ


#Example sanity test for model.fit()

history = model.fit(
    train_generator,
    steps_per_epoch = 16,
    validation_data = valid_generator, 
    validation_steps = 8,
    epochs = 1,
    #callbacks=callbacks_list
    )


4. Evaluate your models. You should have built multiple models, which you should thoroughly evaluate and compare via a robust analysis of residuals and failures.

There are 4 models in this project 

Normal CXR vs Viral CXR
Normal CXR vs Bacterial CXR
Normal CXR vs Viral CXR (Robust)
Normal CXR vs Bacterial CXR (Robust)

There is only a slight difference between the normal model and the Robust model. The code differences will be shown below.

# This is the "regular models" only using shear range and zoom range which can 
#get a fairly tight fitting model in a small number of epoch runs.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    )

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


# This is the ImageDataGenerator for the Robust models.
# rotation_range, width_shift_range, and height_shift_range image transforms
# are added. It can take a very long run time of model.fit() 6 hours or so
# to get a reasonably fit model. The good news is this can accept a much more 
# diverse set of CXR images. (You can accept CXRs from lazy and incompetant
# Xray technicians!) This dataset seemed to be picked from pristine Xray images
# but with variation of images introduced by some children and small adults. 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    )

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



5. Present and thoroughly explain your product. Describe your model in detail: why you chose it, why it works, what problem it solves, how it will run in a production like environment. What would you need to do to maintain it going forward?

The application of these models is very straightforward. The user can input a frontal CXR and get a reasonably reliable answer that the CXR is positive for Viral or Bacterial Penumonia. The implementation will require the image to be tested on both models.
Perhaps a third model in the front to classify the image as CXR or not CXR would be useful in order to prevent a garbage test being performed  

I plan to have the models reside on a flask Django web frontend and I would also like to attempt a android kotlin app that would use the h5 model files or stored as pickle files as a tester for CXR inout. The h5 model files are only 16MB in size. So it should be reasonable to deploy as a mobile application. 

It is certainly not medical grade and would require years of testing to certify for medical diagnostic use.





---



---



---



---



---

Original Project Proposal 

---



---



---



---


1.   What is the problem you are attempting to solve?

     Using machine learning/deep learning to analyze a set of chest xray 
     images and diagnose the presence of lung damage created by covid-19.
     It would also be useful to be able to train a model that would be 
     able to tell the difference between a normal chest x-ray and a 
     chest xray of an pneunomia affected patient. Pneunomia is also a  
     deadly lung disease. 


2.   How is your solution valuable?
     
     An automated tool that could assist a doctor, clinician or nurse 
     and tell them they are looking at a patient that has covid-19 lung   
     dammage is valuable. That patient can be isolated and immediately  
     start treatment. A different set of protocols are used with 
     patients that can be presumed covid-19 positive.  


3.   What is your data source and how will you access it? 

      The data is from 

      https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

      It consists of about 1.2 G of 5935 files of chest xray images. The 
      files have been copied over to my google drive and are accessed 
      from Google collab with access to the google drive mount.



4.   What techniques from the course do you anticipate using?

      The techniques used will be image processing with deep learning and 
      image processing with a CNN deep learning network. I will also try  
      to use Dask either locally on the google collab or with a remote 
      linux distro connected to by a local jupyter notebook running the 
      Dask client. 


5.   What do you anticipate to be the biggest challenge you’ll face? 
     
     The data is limited as far as the covid-19 positive training images. 
     I only have access to 58 of them in this dataset. So far my testing 
     with 3 categories of categorizing a normal dataset versus a 
     pneunomia dataset versus the covid-19 dataset. It would be nice to 
     be able to compare and categorize 1000 normal chest xray images, 
     versus 1000 penunomia xrays versus 1000 covid-19 images. That would 
     make the categorization training fair and not create imabalanced 
     results.    


---





---

---



---



---





---



---



---

