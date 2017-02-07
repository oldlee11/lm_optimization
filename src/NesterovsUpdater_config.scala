package lm

import java.io.Serializable

/**
  * Created by liming on 2017/2/3.
  */
class NesterovsUpdater_config extends Serializable{
  /*
   * 超参数：动量
   * */
  var MomentumSchedule:Option[Double]=None
  def setMomentumSchedule(in_param:Double){
    MomentumSchedule=Some(in_param)
  }
  def getMomentumSchedule():Double={
    MomentumSchedule.get
  }
  def isNoneMomentumSchedule():Boolean={
    MomentumSchedule==None
  }
}
