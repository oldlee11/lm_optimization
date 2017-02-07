package lm

/**
  * Created by liming on 2017/2/4.
  */
class AdamUpdater_config {
  /*
   * 超参数：beta1
   * */
  var beta1:Option[Double]=None
  def setbeta1(in_param:Double){
    beta1=Some(in_param)
  }
  def getbeta1():Double={
    beta1.get
  }
  def isNonebeta1():Boolean={
    beta1==None
  }

  /*
 * 超参数：beta2
 * */
  var beta2:Option[Double]=None
  def setbeta2(in_param:Double){
    beta2=Some(in_param)
  }
  def getbeta2():Double={
    beta2.get
  }
  def isNonebeta2():Boolean={
    beta2==None
  }
}
