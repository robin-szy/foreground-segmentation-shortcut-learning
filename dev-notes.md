### March 17

**3pm**  
YOLO for segmentation of the objects:  
- YOLO seems to need annotation of the objects, which I don't have. So, annotating myself is too much work, so I will need to find another model. SAM seems to be a good option, as it does not need annotation.  
Sources:
https://training.galaxyproject.org/training-material/topics/imaging/tutorials/yolo-segmentation-training/tutorial.html
https://github.com/facebookresearch/segment-anything

**5pm**  
I've tried around a bit, and indeed: SAM does return segmentation masks quite well. However, there are too many, and it can not tell which one I want. So, the semantics are somehow missing. I've tried to create a score which rewards if an object is centered or not, and give some bonus if it is big. However, the results are quite bad.  
This made me think, and I believe I have found a solution: I should build an animal or fungi classifier first (and say that everything else is plantae? Or how to solve this?). Then I will use SAM to return the x best segmentation masks. I will crop the detected objects from these segmentation masks and feed it to the classifier. The one the classifier approves is taken. With this, I create a new dataset with which I can retrain the model. Then I can compare if it got better or not.  
Another alternative could be to annotate 100 images for YOLO from different instances, so it can detect: Fungi or animal, then apply YOLO to segment the images. Then train the model.

**5:35pm**  
Taking back what I just said: SAM might work by itself without my new plan of classifying the masks first. In this blog post https://blogs.torus.ai/segment-anything/, I found that it actually accepts text prompts. But this is not specified in the Github code. So how does it work?

* One thing I found is an open-source library based on SAM: https://github.com/luca-medeiros/lang-segment-anything
* Use CLIP to create a score for each of the segmentation masks: https://docs.ultralytics.com/guides/similarity-search/#advantages-of-semantic-image-search-with-clip-and-faiss

I'll start with CLIP, I guess. This is just to segment the animals from the images. Then, I can train a model from scratch based on the isolated objects. 
