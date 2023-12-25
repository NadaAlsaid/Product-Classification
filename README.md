<h1 align = "center">Product Classification</h1>
Product classification is important task to automate tasks such as inventory management, pricing. For example, a retailer could use this technology to automatically classify products in a store,
and then use this information to update inventory levels, calculatepro ducts prices, and provide with information about the products.<br>
<h2>Data Preparation</h2>

  - Logistic Regression Model & Navie Bayes Model
    - images are retrieved from directory and are saved as an array of grayscale images (.i.e Transformed from colored to be black and white) 

    - apply smoothing filter on images
  
    - The Features are extracted as per image, to generate “bag of features”(bag of Visual words), represented as descriptors and keypoints that created using SIFT
  - CNN Model
    - images are retrieved from directory and are saved as an array of grayscale images
      
    - using smoothing filter then did a resize on images and convert them into a num.array to apply CNN

<h2>Model Description :</h2>

  - Logistic Regression Model
    - The BoV algorithm works with the descriptors of features from multiple images, where each descriptor belongs or describes a certain region as per image.
    - For those descriptors, represented in a high-dimensional space (as vectors), close points indicate similarity, which means that these close descriptor points represent the same feature even though 
      points come from different images.
    - Using K-Means, the close descriptor points in space are grouped/clustered, around a center, this center is known as a “Visual Word”, that provides a representation for all of the points in cluster.
    - Visual Words are grouped to generate Visual Vocabulary, objects in images can be represented and described through the Vocabulary.
    - Then use Logistic Regression classifier

  - Navie Bayes Model
    - It's same as Logistic Regression Model, but using Navie Bayes classifier
  - CNN Model
    - The model architecture consists of several Conv2D layers with different number of filters and kernel sizes, followed by MaxPooling, Flatten, Dense, and Dropout layers. The final layer uses softmax activation for multi-class classification.
    - The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric. It prints the summary of the model.
    - The model is trained using the training data (x_train_new and y_train_new) for a specified number of epochs and batch size.
    - After training, it evaluates the model on the training data and prints the test loss and accuracy
      
<h2>Train Accuracy:</h2>
  - Navie Bayes Model & Logistic Regression Model -> 100%
  - CNN Model -> 77.86 %
<h2>Test Accuracy:</h2>
  - Navie Bayes Model -> 76.47%
  - Logistic Regression Model -> 91.18%
  - CNN Model -> 64.71%
  
# Collaborators:
- <a href="https://github.com/NadaAlsaid">Nada Alsaid</a><br>
- <a href="https://github.com/anna-adel">Yoana Adel</a><br>
- <a href="https://github.com/maHossam9">Mariam Hossam</a><br>
- <a href="https://github.com/NadaShehata">Yara Hossm</a><br>
- <a href="https://github.com/YaraSherif">Yara Sherif</a><br>
- <a href="https://github.com/Yara-Abdelrahem">Yara Abdelrahem</a>
