## paper-sheet-detection

# Project introduction
 This program can scan the paper sheet and transfer it into a **flat rectangle** image by using the information
from **curved side edges**.
# How to use it
  * Put your picture in the same folder with Normal_Warp.py.
  * Replace the name of your picture of paper sheet with 'p8.jpg' in line 7 of Normal_Warp.py
  * Run Normal_Warp
# Requirement
  * python3
  * cv2
  * numba
  * matplotlib
  * numpy
  - - - 
  
  # Backgroud
  The real edge of a document maybe curved rather than straight, so the code will restore a flatter original appearance of the document according to the curved
edge information of the document.

# Effect
  As an example, you can see how my project deal with p8.jpg.
  <div>
<center>
  
<img src="EdgeAndRectangle.png"  alt="Edged" title="Original pic. with marked border" width=230\>
<img src="After - Other - Ratio.png"  alt="After" title="By Quadrilateral way" width=200\>
  <img src="RuledSurface.png" alt="Ruled" title="By Ruled Surface Model(My project)" width=200\>
</center>
</div>
You can see the four-point perspective transformation preserves the curved characteristics of the paper, while the ruled surface restoration of this project flattens the document in the original image to a certain extent.

Therefore, if you has a photo of document, with stright upper and down edges, you can use my program to restore your original flat document.
