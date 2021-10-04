## HotFix Tutorial

Matplotlib returns the wrong index when selecting a point of a 3D scatter plot, the returned index is based on the figure's camera angle rather than the indices of the original scatter array. This is a [know issue](https://stackoverflow.com/questions/66442220/matplotlib-scatter-3d-button-press-event-give-wrong-index-based-on-scatter-axis) since matplotlib is really not a 3D figure package, but it can be hot fixed. 
In order to fix this, a function inside a toolkit must be changed so that the user can obtain the protected original indices. The link provides a quick fix, but we used another working fix.


## Steps

* Open the Python script results_multiobjective.py archivos
   [results_multiobjective.py](results_multiobjective.py)
* Ctrl + Click on the art3D imported module of the mpl_toolkits Library. Some IDEs also allow Ctrl + B.

```python
import mpl_toolkits.mplot3d.art3d
```

* Add `self.` in front of each one of the 6 `z_markers_idx` object inside the `do_3d_projection` function of the `Path3DCollection` class (lines 605-622 matplotlib 3.4.3).
* art3d.py script should look like this ![Fixed MPL Module.png](Fixed%20MPL%20Module.png)
* The [results_multiobjective.py](results_multiobjective.py) script will work as intended.
* To roll back, reinstall matplotlib.