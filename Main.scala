import spark.SparkContext
import spark.SparkContext._
import scala.io.Source

object ML {
  def main(args: Array[String]) {
    val sparkHome = "/root/spark"
    val jarFile = "target/scala-2.9.2/scala-app-template_2.9.2-0.0.jar"
    val sc = new SparkContext("local", "TestJob", sparkHome, Seq(jarFile))
    // println("1+2+...+10 = " + sc.parallelize(1 to 10).reduce(_ + _))

    var dta = List[Array[Double]]()
    var resp = List[Double]()
    /* read the data in, and append it to the list, by returning a new list */
    for(val line <- Source.fromFile("data_file").getLines) {
      var l = line.split(" ")
      dta = dta ::: List(List(l(0).toDouble, l(1).toDouble).toArray)
      resp = resp :+ l(2).toDouble // append a double to the end of the list
    }

    /* dump the data we are using to the console */
    dta.foreach(deep_print)
    println("\n\n")
    resp.foreach(println)

    /* parallelize the data, and count the length */
    // var data = sc.parallelize(lst)
    // println(data.count())

    /* test dot_product function */
    val a = List(1.0, 2.0, 3.0, 4.0, 5.0)
    val b = List(6.0, 7.0, 8.0, 9.0, 10.0)
    if (dot_product(a.toArray, b.toArray) == 130) {
      println("dot_product passed tests")
    } else {
      println("dot_product failed tests")
    }

    /* test vec_subtract function */
    if (vec_subtract(a.toArray, b.toArray)(0) == -5) {
      println("vec_subtract passed tests")
    } else {
      println("vec_subtract failed tests")
    }

    /* test obj_fn function */
    val c = List(List(1.0, 1.0).toArray, List(1.0, 1.0).toArray) // data matrix
    val d = List(2.0, 2.0).toArray // weight vector
    val e = List(4.0, 4.0).toArray // response vector
    if (obj_fn(c,d,e) == 0) {
      println("obj_fn passed tests")
    } else {
      println("obj_fn failed tests")
    }

    /* test grad function */
    if (grad(c,d,e)(0) == 0) {
      println("grad passed tests")
    } else {
      println("grad failed tests")
    }

    /* test vec_scale function */
    if (vec_scale(List(1.0, 1.0).toArray, 10.0)(0) == 10) {
      println("vec_scale function passed tests")
    } else {
      println("vec_scale function failed tests")
    }

    /* begin optimization - CONVERGES TO SOLUTION */
    val maxiter = 250
    val alpha = .0001
    var x = List(-10.0, -10.0).toArray
    for (i <- 0 until maxiter) {
      val g = grad(dta, x, resp.toArray) // compute gradient at x
      x = vec_subtract(x, vec_scale(g, alpha))
      print("weight vector = ")
      deep_print(x)
    }
    println("\ncompleted optimization succesfully\n")

  }

  /* gradient function nabla_x_i = (y_i - a_i * x)a_ij */
  def grad(a: List[Array[Double]], x: Array[Double], b: Array[Double]) : Array[Double] = {
    var gradient = new Array[Double](x.length)
    val x_length = x.length
    val a_length = a.length
    for (j <- 0 until x_length) {  // loop over x_j
      for (i <- 0 until a_length) { // loop over a_i's and b_i's to find residual
        val dp = dot_product(a(i), x)
        gradient(j) += (dp - b(i)) * a(i)(j)
      }
    }
    gradient
  }

  /* linear regression objective function: .5||Ax-b||_2^2 */
  def obj_fn(a: List[Array[Double]], x: Array[Double], b: Array[Double]) : Double = {
    var accumulator = 0.0
    val len = a.length
    for (i <- 0 until len) {
      val dp = dot_product(a(i), x)
      accumulator += .5 * (dp - b(i)) * (dp - b(i))
    }
    accumulator
  }

  /* compute the dot-product of two vectors - scalar result */
  def dot_product(a: Array[Double], b: Array[Double]) : Double = {
    return a.zip(b).map{case (a,b) => a * b}.reduceLeft(_ + _)
  }

  /* compute the difference of two vectors */
  def vec_subtract(a: Array[Double], b: Array[Double]) : Array[Double] = {
    return a.zip(b).map{case (a,b) => a - b}
  }

  /* deep-print an array with a.length elements */
  def deep_print(a: Array[Double]) = {
    val a_length = a.length
    for (i <- 0 until a_length) {
        print(a(i) + " ")
    }
    print("\n")
  }

  /* scale a vector - purity violation */
  def vec_scale(a: Array[Double], alpha: Double) : Array[Double] = {
    val a_length = a.length
    for (i <- 0 until a_length) {
      a(i) = alpha * a(i)
    }
    a
  }
}