
# NumSca: NumPy for Scala

NumSca is NumPy for Scala.
For example, here's the famous [neural network in 11 lines of python code](http://iamtrask.github.io/2015/07/12/basic-python-network/), this time in scala:

```scala
val x = ns.array( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1).reshape(4, 3)
val y = ns.array(0, 1, 1, 0).T
val w0 = 2 * ns.rand(3, 4) - 1
val w1 = 2 * ns.rand(4, 1) - 1
for (j <- 0 until 60000) {
  val l1 = 1 / (1 + ns.exp(-ns.dot(x, w0)))
  val l2 = 1 / (1 + ns.exp(-ns.dot(l1, w1)))
  val l2_error = ns.mean(ns.abs(y - l2)).squeeze()
  val l2_delta = (y - l2) * (l2 * (1 - l2))
  val l1_delta = l2_delta.dot(w1.T) * (l1 * (1 - l1))
  w1 += l1.T.dot(l2_delta)
  w0 += x.T.dot(l1_delta)
}
``` 

## Importing numsca
```scala
import botkop.{numsca => ns}
```

## Creating a Tensor

```scala
scala> Tensor(3,2,1,0)
[3.00,  2.00,  1.00,  0.00]

scala> ns.zeros(3, 3)
[[0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00]]

scala> ns.ones(3, 2)
[[1.00,  1.00],
 [1.00,  1.00],
 [1.00,  1.00]]
 
scala> val ta: Tensor = ns.arange(10)
[0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]
 
 scala> val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)
 [[[0.00,  1.00,  2.00,  3.00],
   [4.00,  5.00,  6.00,  7.00],
   [8.00,  9.00,  10.00,  11.00]],
 
  [[12.00,  13.00,  14.00,  15.00],
   [16.00,  17.00,  18.00,  19.00],
   [20.00,  21.00,  22.00,  23.00]]]
```

## Access
Single element
```scala
scala> ta(0)
res10: botkop.numsca.Tensor = 0.00

scala> tc(0, 1, 2)
res14: botkop.numsca.Tensor = 6.00
```
Get the value of a single element Tensor:
```scala
scala> ta(0).squeeze()
res11: Double = 0.0
```
Slice
```scala
scala> tc(0)
res7: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00,  3.00],
 [4.00,  5.00,  6.00,  7.00],
 [8.00,  9.00,  10.00,  11.00]]
 
scala> tc(0, 1)
res8: botkop.numsca.Tensor = [4.00,  5.00,  6.00,  7.00]
```

## Update
In place
```scala
scala> val t = ta.copy()
t: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(3) := -5
scala> t
res16: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(0) += 7
scala> t
res18: botkop.numsca.Tensor = [7.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```

Array wise
```scala
scala> val a2 = 2 * ta
val a2 = 2 * ta
a2: botkop.numsca.Tensor = [0.00,  2.00,  4.00,  6.00,  8.00,  10.00,  12.00,  14.00,  16.00,  18.00]
```

## Slicing
Note: 
- step size is not implemented.
- python notation ```t[:3]``` must be written as ```t(0 :> 3)``` or ```t(:>(3))``` 
- negative indexing is supported
- ellipsis is not implemented

### Single dimension
#### Slice over a single dimension

```scala
scala> val a0 = ta.copy().reshape(10, 1)
a0: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a1 = a0(1 :>)
a1: botkop.numsca.Tensor = [1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a2 = a0(0 :> -1)
a2: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00]

scala> val a3 = a1 - a2
a3: botkop.numsca.Tensor = [1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00]

scala> ta(:>, 5 :>)
res19: botkop.numsca.Tensor = [5.00,  6.00,  7.00,  8.00,  9.00]

scala> ta(:>, -3 :>)
res4: botkop.numsca.Tensor = [7.00,  8.00,  9.00]
```

#### Update single dimension slice

```scala
scala> val t = ta.copy()
t: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Assign another tensor
```scala
scala> t(2 :> 5) := -ns.ones(3)
scala> t
res6: botkop.numsca.Tensor = [0.00,  1.00,  -1.00,  -1.00,  -1.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Assign a value
```scala
scala> t(2 :> 5) := 33
scala> t
res8: botkop.numsca.Tensor = [0.00,  1.00,  33.00,  33.00,  33.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Update in place
```scala
scala> t(2 :> 5) -= 1
scala> t
res10: botkop.numsca.Tensor = [0.00,  1.00,  32.00,  32.00,  32.00,  5.00,  6.00,  7.00,  8.00,  9.00]

```

### Multidimensional slices
```scala
scala> tb
res11: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]
 
scala> tb(2:>, :>)
res15: botkop.numsca.Tensor = [6.00,  7.00,  8.00]
```
Mixed range/integer indexing. Note that integers are implicitly translated to ranges, 
and this differs from python. 
```scala
scala> tb(1, 0 :> -1)
res1: botkop.numsca.Tensor = [3.00,  4.00]
```

## Fancy indexing
### Boolean indexing
```scala
scala> val c = ta < 5 && ta > 1
c: botkop.numsca.Tensor = [0.00,  0.00,  1.00,  1.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00]
```
This returns a TensorSelection:
```scala
scala> val d = ta(c)
d: botkop.numsca.TensorSelection = TensorSelection([0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00],[[I@153ea1aa,None)
```
Which is implicitly converted to a Tensor when needed:
```scala
scala> val d: Tensor = ta(c)
d: botkop.numsca.Tensor = [2.00,  3.00,  4.00]
```
Or you can force it to become a Tensor:
```scala
scala> ta(c).asTensor
res10: botkop.numsca.Tensor = [2.00,  3.00,  4.00]
```
Updating:
```scala
scala> val t = ta.copy()
scala> t(ta < 5 && ta > 1) := -7
res6: botkop.numsca.Tensor = [0.00,  1.00,  -7.00,  -7.00,  -7.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```
Selection over multiple dimensions:
```scala
scala> val c: Tensor = tc(tc % 5 == 0)
c: botkop.numsca.Tensor = [0.00,  5.00,  10.00,  15.00,  20.00]
```
Updating over multiple dimensions:
```scala
scala> val t1 = tc.copy()
t1: botkop.numsca.Tensor =
[[[0.00,  1.00,  2.00,  3.00],
  [4.00,  5.00,  6.00,  7.00],
  [8.00,  9.00,  10.00,  11.00]],

 [[12.00,  13.00,  14.00,  15.00],
  [16.00,  17.00,  18.00,  19.00],
  [20.00,  21.00,  22.00,  23.00]]]
  
scala> t1(t1 > 5 && t1 < 15) *= 2
res21: botkop.numsca.Tensor =
[[[0.00,  1.00,  2.00,  3.00],
  [4.00,  5.00,  12.00,  14.00],
  [16.00,  18.00,  20.00,  22.00]],

 [[24.00,  26.00,  28.00,  15.00],
  [16.00,  17.00,  18.00,  19.00],
  [20.00,  21.00,  22.00,  23.00]]]
```
### List of location indexing
```scala
scala> val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)

scala> val idx = Tensor(3, 4, 1, 2, 2)

scala> primes(idx).asTensor
res23: botkop.numsca.Tensor = [7.00,  11.00,  3.00,  5.00,  5.00]

```
Reshape according to index:
```scala
scala> tb
res25: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]

scala> primes(tb).asTensor
res24: botkop.numsca.Tensor =
[[2.00,  3.00,  5.00],
 [7.00,  11.00,  13.00],
 [17.00,  19.00,  23.00]]
```
Use as a look-up table:
```scala
scala> val numSamples = 4
       val numClasses = 3
       val x = ns.arange(numSamples * numClasses).reshape(numSamples, numClasses)
       val y = Tensor(0, 1, 2, 1)
       val z: Tensor = x(ns.arange(numSamples), y)
res26: botkop.numsca.Tensor = [0.00,  4.00,  8.00,  10.00]
```
Update along a single dimension:
```scala
scala> val primes = Tensor(2, 3, 5, 7, 11, 13, 17, 19, 23)
primes: botkop.numsca.Tensor = [2.00,  3.00,  5.00,  7.00,  11.00,  13.00,  17.00,  19.00,  23.00]

scala> val idx = Tensor(3, 4, 1, 2, 2)
idx: botkop.numsca.Tensor = [3.00,  4.00,  1.00,  2.00,  2.00]

scala> primes(idx) := 0

scala> primes
res1: botkop.numsca.Tensor = [2.00,  0.00,  0.00,  0.00,  0.00,  13.00,  17.00,  19.00,  23.00]
```
Multiple dimensions
```scala

```

