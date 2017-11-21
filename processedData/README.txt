Each of the files x,y, and z contained here are formatted for the convenience of the user.  

Each of the four files contain 7037 lines.  Each line is a data sample using Marcus Lim's feature extraction along with Kerekes and Meusling's data and labels.  The line numbers map to a particular labelled rectangle.  So line 1 in each file refers to the first labelled rectangle, and line 200 in each file refers to the 200th labelled rectangle for example.

features
This file is formatted with rewards and features combined (x and y) and is already in the proper format to plug into SVM-Light.

x.txt
Each line has 1901 space-delimitted floating point values.  Line 1 is the first labelled rectangle's 1901 extracted features.  Line 7037 is the final labelled rectangle's 1901 extracted features.  Each line corresponds to the same line in y.txt and z.txt.


y.txt
Each line has the associated reward for each sample.  Line 1 has the reward for the first labelled rectangle.  Line 7037 has the reward for the final labelled rectangle.  For our purposes, each reward is either +1 or -1, meaning "good grasping rectangle" or "bad grasping rectangle."

z.txt
Each line has four space-delimitted pieces of data.  First is the image id that the rectangle is taken from (0000 through 1034).  Second is the object id (0 through 281), since most objects have multiple images.  Each object id represents a different object.  Three different bowls will have three different object ids.  Third is a short description of what the item is.  Fourth is the identifier for which background image to use if you wish to perform background subtraction.  The background image may or may not be useful for you depending on how you plan to identify the object to grasp in the image.
