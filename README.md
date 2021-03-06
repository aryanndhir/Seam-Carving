# Seam-Carving

Used openCV in Python for processing images and applied Dynamic Programming to create energy maps.

Seam carving uses an energy function defining the importance of pixels. A seam is a connected path of low energy pixels crossing the image from top to bottom, or from left to right. By successively removing or inserting seams we can reduce, as well as enlarge, the size of an image in both directions. 

For image reduction, seam selection ensures that while preserving the image structure, we remove more of the low energy pixels and fewer of the high energy ones. 

For image enlarging, the order of seam insertion ensures a balance between the original image content and the artificially inserted pixels. These operators produce, in effect, a content-aware resizing of images. 
