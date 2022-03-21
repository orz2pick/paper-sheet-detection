# paper-sheet-detection

This code can scan a curved edge paper and transfer it into a rectangle sheet.

Let's consider following document.
<img src="https://b3logfile.com/siyuan/1610205759005/assets/image-20210815154125-whdthe1.png" alt="图片alt" title="图片title">

And
<center>

        <img style="border - radius:  0.3125em;

        box - shadow : 0 2px 4px 0 rgba(34, 36,38,.12), 0 2px 10px 0 rgba(34,36,38,.08);"

        src="p8.jpg">

        <br>

        <div style="color:orange; border - bottom: 1px solid #d9d9d9;

         display: inline-block;

          color: #999;

          padding: 2px;">Original Image</div>

</center>

Its edge is curved. Usually the software(IOS, Android) would use circumscribed quadrilateral to approximate the border, and then use warp perspection to transfer the image to a rectangle. This kind of effect is following:

<center>

<figure>
<img src="EdgeAndRectangle.png" />
<img src="After.png" />
</figure>



<\center>
