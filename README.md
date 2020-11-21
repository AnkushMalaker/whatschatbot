## Model Description

## Training 

## Usage



## Requirements 


## Notes
1.    It's important to note that chat conversations between people are much different than more formal text corpuses and offer
some special challenges.
-    The interlap of images and text, emojis and stickers making the data incomplete if extracted without those elements
-    If those elements are extracted, making them so sparse in comparison to the rest of the training data that it's hard to fit the model, not to forget, the difficulty in accounting for the difference in modality.

2.    The amount of training data that one may achieve is highly dependant on the strictness of the preprocessing. 
-    Say we remove all instances where the reply was less than 4 characters, we remove all instances where a user says "ok", which from preliminary text analysis is very high in frequency. There may be many other such examples. Finding the right balance between availability of data and quality will be crucial for this as the length and properties of every chat corpus will differ.

## Work remaining
-    Update readme
-    Make load_weights and evaluation scripts
-    Get better data and train model 
