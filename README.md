# Modules

 * ml-hypersim  
 * prepare patches
 * CNN 

# TODO/to be improved

### Simple TODO (possibly redundant) 
 * !!stack expects each tensor to be equal size, but got [3, 11, 11] at entry 0 and [3, 7, 7] at entry 1
   * probably problem of stack, still it's a pain IMHO - ask 
 * warnings, TODOS in code
 * wandb
   * account and make it work (DONE)
   * explore
 * scale of the patch (DONE)
 * correspondence of the scales !!
 * augment data (how!!)
 * possibly check and/or estimate the dominant direction as well
 * detector can be disk or superpoint
 * even higher precision via 
   * overlapping imgs
   * 3D geometry
 * CNN: (batch) normalization
 * SIFT/DoG location read up
 * run on a bigger data (!)

## Data


 * input in general (derivations in scale space)
 * scale of the patch (!) - DONE
 * when to deem the locations from original img as GT? 
    * threshold on minimal scale? - yes (+ correspondence of the scales)
 * augment data (rotation); careful: (lanczos!)
 
## CNN

 * baseline model (Resnet) - postpone (speed?)
 * (batch) normalization
 * own custom architecture? 
 * others... 

## Others

* detector descriptor dependence
  * train together (?) 
  * the dependence is the same in all the matching pipelines btw. 

* would like to analyze the way DoG locations is computed (quadratic interpolation)
  * e.g. does is mean there is a given upper bound on the error 
    * consequences for the regression?

* next?
  * implement what has been suggested
  * try on a bigger data with a bigger validation set
    * downstream task - not yet!
    
