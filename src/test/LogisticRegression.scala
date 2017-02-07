package lm.test

/**
  * Created by liming on 2017/2/4.
  */
import lm.{BaseUpdater, SgdUpdater}

import scala.math.log    //用于建立可变的array

class LogisticRegression(n_in: Int,
                          n_out: Int,
                          updater:BaseUpdater) {
  val W: Array[Array[Double]] = Array.ofDim[Double](n_out, n_in+1)//初始化为了0.0 注意b直接放于w内
  var W_flatten=W.flatten
  var cross_entropy_result:Double=0.0

  //根据一个样本计算梯度(n_out * n_in+1)
  def train_paramdiff_onesample(x: Array[Double], y: Array[Int]):Array[Array[Double]] ={
    val p_y_given_x: Array[Double] = new Array[Double](n_out)
    //正向
    for(i <- 0 until n_out) {
      p_y_given_x(i) = 0
      for(j <- 0 to n_in) {
        if(j!=n_in){
          //j<-[0,n_in-1]
          p_y_given_x(i) += W(i)(j) * x(j)
        }else{
          //j= n_in
          p_y_given_x(i) += W(i)(j) * 1
        }
      }
    }
    val p_y_given_x_softmax:Array[Double]=softmax(p_y_given_x)//相当于p_y_given_x_softmax=softmax(p_y_given_x)
    this.cross_entropy_result=this.cross_entropy_result + cross_entropy(y.map(x=>x.toDouble),p_y_given_x_softmax)
    val dy: Array[Double] = new Array[Double](n_out)
    for(i <- 0 until n_out) {
      dy(i) = p_y_given_x_softmax(i) - y(i)
    }
    for{
      dy_i<-dy
    }yield (x :+ 1.0).map(x_i=>x_i*dy_i)
  }

  //根据一批样本更新参数
  def train(x:Array[Array[Double]],y:Array[Array[Int]],iteration:Int){
    cross_entropy_result=0.0
    val gradient_org_batch=(for{
      i <- (0 until x.length)
    }yield train_paramdiff_onesample(x(i),y(i)).flatten).toArray
    W_flatten=W.flatten
    updater.update(W_flatten,gradient_org_batch,iteration)
    for(i <- 0 until n_out){
      for(j <- 0 to n_in){
        W(i)(j)=W_flatten(i*(n_in+1)+j)
      }
    }
    println("cross_entropy="+cross_entropy_result/x.length)
  }

  def softmax(x: Array[Double]):Array[Double]= {
    val result:Array[Double]=new Array(n_out)
    var max: Double = 0.0
    var sum: Double = 0.0
    var i: Int = 0
    for(i <- 0 until n_out) if(max < x(i)) max = x(i)
    for(i <- 0 until n_out) {
      result(i) = math.exp(x(i) - max)
      sum += result(i)
    }
    for(i <- 0 until n_out) result(i)=result(i)/sum
    result
  }

  def predict(x: Array[Double]):Array[Double]={
    var i: Int = 0
    var j: Int = 0
    val tmp:Array[Double]=new Array(n_out)
    for(i <- 0 until n_out) {
      tmp(i) = 0
      for(j <- 0 to n_in) {
        if(j!=n_in){
          tmp(i) += W(i)(j) * x(j)
        }else{
          tmp(i) +=W(i)(j) *1
        }
      }
    }
    softmax(tmp)
  }

  //使用交叉信息嫡cross-entropy衡量样本输入和经过编解码后输出的相近程度
  //值在>=0 当为0时 表示距离接近
  //每次批量迭代后该值越来越小
  def cross_entropy(x: Array[Double], z: Array[Double]):Double={
    -1.0* (0 until x.length).map(i=>x(i)*log(z(i))+(1-x(i))*log(1-z(i))).reduce(_+_)
  }

  def init_w_module(){
    //w
    for(i <- 0 until n_out)
      for(j <- 0 to n_in)
        W(i)(j) = 0.0
  }

}

object LogisticRegression {
  def test_LR_simple() {
    val n_epochs: Int = 500
    val n_in: Int = 9
    val n_out: Int = 3
    val train_X: Array[Array[Double]] = Array(
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(1, 0, 1, 0, 0, 0,0,0,0),
      Array(1, 1, 1, 0, 0, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 1, 0, 1, 0,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,0),
      Array(0, 0, 0, 0, 0, 0,1, 1, 1),
      Array(0, 0, 1, 0, 0, 0,1, 1, 1),
      Array(1, 0, 0, 0, 0, 0,1, 0, 1)
    )
    val train_Y: Array[Array[Int]] = Array(
      Array(1, 0,0),
      Array(1, 0,0),
      Array(1, 0,0),
      Array(0, 1,0),
      Array(0, 1,0),
      Array(0, 1,0),
      Array(0, 0,1),
      Array(0, 0,1),
      Array(0, 0,1)
    )
    val train_N=train_X.length

    val updater=new SgdUpdater()
    updater.config.setLearningRateByParam(0.9)
    updater.config.setisMiniBatch(true)
    updater.config.setminiBatchSize(train_N)
    updater.config.setLearningRatePolicy("Exponential")
    updater.config.setLrPolicyDecayRate(1)
    updater.config.setL2ByParam(0.01)

    // construct
    val classifier = new LogisticRegression(n_in,n_out,updater)
    // train
    var epoch: Int = 0
    var i: Int = 0
    for(epoch <- 0 until n_epochs) {
      classifier.train(train_X, train_Y,epoch)
    }
    // test data
    val test_X: Array[Array[Double]] = Array(
      Array(1, 0, 1, 0, 0, 1,0,0,0),
      Array(0, 0, 1, 1, 1, 0,0,0,1),
      Array(0, 0, 0, 0, 1, 0,1,1,1)
    )
    val test_N=test_X.length
    val test_Y: Array[Array[Double]] = Array.ofDim[Double](test_N, n_out)
    // test
    var j: Int = 0
    for(i <- 0 until test_N) {
      test_Y(i)=classifier.predict(test_X(i))
      for(j <- 0 until n_out) {
        printf("%.5f ", test_Y(i)(j))
      }
      println()
      /*
0.88771 0.06374 0.04855
0.01545 0.91327 0.07127
0.00273 0.08178 0.91549
       * */
    }
  }

  def main(args: Array[String]) {
    test_LR_simple()
  }

}

