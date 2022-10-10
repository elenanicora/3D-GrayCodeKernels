# 3D Gray-Code Kernels
This demo implements the efficient convolution of a video clip with a family of 3D Gray-Code Kernels.

File binary_tree is responsible for the creation of a family of 3D Gray-Code Kernels of dimension 4x4x4 and their re-ordering. A function for the visualisation of each kernel is available.

File efficient_scheme applies the efficient convolution by computing a first full fft-conv (with the first kernel of the family) and then obtains the projections (with the remaining kernels of the family) by applying an efficient projection scheme that requires a constant number of operations per pixel, regardless the kernel size.

File 3DGCK_onframes calls the above functions on overlapping clips of video (stride=1). The number of frames considered for each clip is related to the dimension of the kernels (in this demo 4 consecutive frames).

Users can either visualise, save the entire bank of projections per clip or save a pooled, more compact, version of the projections that will give hints on motion saliency properties found within the video.

# Reference
Nicora, Elena, and Nicoletta Noceti. "On the use of efficient projection kernels for motion-based visual saliency estimation." Frontiers in Computer Science: 67.

https://www.frontiersin.org/articles/10.3389/fcomp.2022.867289/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Computer_Science&id=867289
