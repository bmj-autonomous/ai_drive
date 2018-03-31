
Links
^^^^^^^^^^^^^^^^^^^^^^^^^
YES: ![car](https://marcusjones.github.io/ai.drive/Post1_2018FEB02/Car1.jpg)

OR, replace base url with /, as in: 

YES ![car](/Post1_2018FEB02/Car1.jpg)

YES ![car](/ai.drive/Post1_2018FEB02/Car1.jpg)

The following DON'T work:

NO: ![car](https://github.com/MarcusJones/ai.drive/blob/master/docs/Post1_2018FEB02/Car1.jpg)

NO: ![car](https://github.com/MarcusJones/ai.drive/blob/master/docs/Post1_2018FEB02/Car1.jpg)


NO: ![car](https://marcusjones.github.io/ai.drive/docs/Post1_2018FEB02/Car1.jpg)

NO: ![car](https://marcusjones.github.io/ai.drive/blob/master/docs/Post1_2018FEB02/Car1.jpg)


INLINE HTML
^^^^^^^^^^^^^^^^^^^^^^^^^
YES WORKS:


{::nomarkdown}<i>foo</i>
<img src="https://marcusjones.github.io/ai.drive/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" height = 100px width="48">
{:/} 


NO: 
{::nomarkdown}<i>foo</i>
<img src="/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" width="48">
{:/} 

{::nomarkdown}<i>foo</i>
<img src="/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" height = 100px width="48">
{:/} 


