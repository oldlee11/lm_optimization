package lm

import java.io.Serializable

/**
  * Created by liming on 2017/2/3.
  */
class BaseUpdater_config extends Serializable{

  /**
    * 是否使用批量模式
    */
  var isMiniBatch:Boolean=true
  def setisMiniBatch(in_param:Boolean){
    isMiniBatch=in_param
  }
  def getisMiniBatch():Boolean={
    isMiniBatch
  }
  /*
   * 一次参数迭代时,使用的样本量
   * */
  var miniBatchSize:Option[Int]=None
  def setminiBatchSize(in_param:Int){
    miniBatchSize=Some(in_param)
  }
  def getminiBatchSize():Int={
    miniBatchSize.get
  }
  def isNoneminiBatchSize():Boolean={
    miniBatchSize==None
  }

  /*
  * 超参数：学习率(步长)
  * */
  var LearningRateByParam:Option[Double]=None
  def setLearningRateByParam(in_param:Double){
    LearningRateByParam=Some(in_param)
  }
  def getLearningRateByParam():Double={
    LearningRateByParam.get
  }
  def isNoneLearningRateByParam():Boolean={
    LearningRateByParam==None
  }

  //学习率的更新策略
  //详见BaseUpdater的applyLrDecayPolicy函数
  var LearningRatePolicy:Option[String]=None
  def setLearningRatePolicy(in_param:String){
    LearningRatePolicy=Some(in_param)
  }
  def getLearningRatePolicy():String={
    LearningRatePolicy.get
  }
  def isNoneLearningRatePolicy():Boolean={
    //如果LearningRatePolicy 设置错误或者没有设置 则返回true
    if(LearningRatePolicy==None){
      true
    }else{
      !Set("Exponential","Inverse","Step","Sigmoid","NoChange").contains(LearningRatePolicy.get)
    }
  }
  //学习率的更新策略 所使用的参数1
  //详见BaseUpdater的applyLrDecayPolicy函数
  var LrPolicyDecayRate:Option[Double]=None
  def setLrPolicyDecayRate(in_param:Double){
    LrPolicyDecayRate=Some(in_param)
  }
  def getLrPolicyDecayRate():Double={
    LrPolicyDecayRate.get
  }
  def isNoneLrPolicyDecayRate():Boolean={
    LrPolicyDecayRate==None
  }
  //学习率的更新策略 所使用的参数2
  //详见BaseUpdater的applyLrDecayPolicy函数
  var LrPolicyPower:Option[Double]=None
  def setLrPolicyPower(in_param:Double){
    LrPolicyPower=Some(in_param)
  }
  def getLrPolicyPower():Double={
    LrPolicyPower.get
  }
  def isNoneLrPolicyPower():Boolean={
    LrPolicyPower==None
  }
  //学习率的更新策略 所使用的参数3
  //详见BaseUpdater的applyLrDecayPolicy函数
  var LrPolicySteps:Option[Double]=None
  def setLrPolicySteps(in_param:Double){
    LrPolicySteps=Some(in_param)
  }
  def getLrPolicySteps():Double={
    LrPolicySteps.get
  }
  def isNoneLrPolicySteps():Boolean={
    LrPolicySteps==None
  }

  /*
  * 超参数：L1型正则化参数
  * */
  var L1ByParam:Option[Double]=None
  def setL1ByParam(in_param:Double){
    L1ByParam=Some(in_param)
  }
  def getL1ByParam():Double={
    L1ByParam.get
  }
  def isNoneL1ByParam():Boolean={
    L1ByParam==None
  }

  /*
   * 超参数：L2型正则化参数
  * */
  var L2ByParam:Option[Double]=None
  def setL2ByParam(in_param:Double){
    L2ByParam=Some(in_param)
  }
  def getL2ByParam():Double={
    L2ByParam.get
  }
  def isNoneL2ByParam():Boolean={
    L2ByParam==None
  }

}

/*
object BaseUpdater_config{
  def main(args: Array[String]) {
    val test_config:BaseUpdater_config=new BaseUpdater_config()
    test_config.setLearningRateByParam(0.5)
    test_config.setminiBatchSize(100)
    test_config.setL2ByParam(0.1)
    println(test_config.isNoneLearningRateByParam())
    if(!test_config.isNoneL1ByParam()) println(test_config.getL1ByParam())
    if(!test_config.isNoneL2ByParam()) println(test_config.getL2ByParam())
  }
}
*/

