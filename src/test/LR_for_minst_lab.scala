package lm.test

import lm.BaseUpdater

/**
  * Created by liming on 2017/2/4.
  */
class LR_for_minst_lab(n_epochs:Int,
                        updater:BaseUpdater) {
  /*
  初始化模型
  */
  val n_ins: Int = 28*28
  val n_outs: Int = 10
  val classifier = new LogisticRegression(n_ins,n_outs,updater)

  def main_deal(): Unit ={
    /*
   train
   */
    val filePath_train:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/valid_data.txt"//train数据量大,暂时使用valid数据
    val train_X:Array[Array[Double]]=dataset.load_mnist(filePath_train).map(x=>x._2)
    //例如    0---->Array(1,0,0,0,0,0,0,0,0,0)
    //        9---->Array(0,0,0,0,0,0,0,0,0,1)
    def trans_int_to_bin(int_in:Int):Array[Int]={
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(int_in)=1
      result
    }
    val train_Y:Array[Array[Int]]=dataset.load_mnist(filePath_train).map(x=>trans_int_to_bin(x._1))
    val train_N: Int = train_X.length
    var epoch: Int = 0
    for(epoch <- 0 until n_epochs) {
      print("epoch_"+epoch+":\n")
      classifier.train(train_X, train_Y,epoch)
    }

    /*
    test
    */
    val filePath_test:String="D:/youku_work/python/spark_python_scala/scala/workpace/deeplearning/dataset/mnist/test_data.txt"
    val test_X:Array[Array[Double]]=dataset.load_mnist(filePath_test).map(x=>x._2)
    val test_N: Int = test_X.length
    val test_Y_pred: Array[Array[Double]] = Array.ofDim[Double](test_N, n_outs)
    val test_Y: Array[Array[Int]]=dataset.load_mnist(filePath_test).map(x=>trans_int_to_bin(x._1))

    //预测并对比
    //Array(0.1191,0.7101,0.0012)-->Array(0,1,0)
    def trans_pred_to_bin(pred_in:Array[Double]):Array[Int]={
      var max_index:Int=0
      for(i <-1 until pred_in.length){
        if(pred_in(i)>pred_in(max_index)){
          max_index=i
        }
      }
      val result:Array[Int]=Array(0,0,0,0,0,0,0,0,0,0);
      result(max_index)=1
      result
    }

    //Array(1,0,0,0,0,0,0,0,0,0)---->0
    def trans_bin_to_int(bin_in:Array[Int]):Int={
      var result:Int= -1;
      for(i <-0 until bin_in.length){
        if(bin_in(i)==1){
          result=i
        }
      }
      result
    }

    //test
    var pred_right_nums:Int=0;
    for(i <- 0 until test_N) {
      test_Y_pred(i)=classifier.predict(test_X(i))
      print("第"+i+"个样本实际值:\n")
      print(test_Y(i).mkString(sep=","))
      print("\n")
      print("第"+i+"个样本预测值:\n")
      print(test_Y_pred(i).mkString(sep=","))
      print("\n")
      if(trans_bin_to_int(trans_pred_to_bin(test_Y_pred(i)))==trans_bin_to_int(test_Y(i))){
        pred_right_nums +=1
      }
    }
    println(pred_right_nums.toDouble/(test_N.toDouble))

  }
}

