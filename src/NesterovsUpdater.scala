package lm

/**
  * Created by liming on 2017/2/4.
  */
class NesterovsUpdater extends BaseUpdater{
  var vPrev:Array[Double]=Array()
  var vPrev_tmp:Array[Double]=Array()
  val config_Nesterovs:NesterovsUpdater_config=new NesterovsUpdater_config()

  override def update(paramVal:Array[Double],
                        gradient_org_batch:Array[Array[Double]],
                        iteration:Int):Boolean ={
    try{
      val paramVallen:Int=paramVal.length
      if(paramVallen!=gradient_org_batch(0).length){
        false
      }else{
        //使用一批样本产生的梯度gradient_org_batch,计算系数的变化量
        //每个样本产生的系数变化量相加=批量的变化量
        val ParamDiff:Array[Double]=gradient_org_batch.
          map(x=> gen_ParamDiff(paramVal=paramVal,
            gradient_org=x,
            iteration=iteration)).
          reduce((x,y)=>x.zip(y).map(x=>x._1+x._2))
        (0 until paramVallen).map(index=>paramVal(index)=paramVal(index)-ParamDiff(index))
        vPrev=vPrev_tmp.map(x=>x/config.getminiBatchSize())
        vPrev_tmp=Array()
        true
      }
    }catch {
      case ex: Exception => false
    }
  }

  override def getGradient(gradient:Array[Double], iteration:Int):Array[Double]={
    /*
    * v=gradient*LearningRateByParam-vPrev*MomentumSchedule
    * 参数变化量Param_Diff=v
    * 注意在批量模式下 vPrev 是一批样本的v的均值,在getGradient函数内做累加,在update函数内做平均
    * */
    val v:Array[Double]=if(vPrev.length==0){
      gradient.map(x=>x*config.getLearningRateByParam())
    }else{
      gradient.map(x=>x*config.getLearningRateByParam()).
        zip(vPrev.map(x=>x*config_Nesterovs.getMomentumSchedule())).
        map(x=>x._1-x._2)
    }
    //更新vprev(用于批量模式)
    if(vPrev_tmp.length==0){
      vPrev_tmp=v
    }else{
      vPrev_tmp=vPrev_tmp.zip(v).map(x=>x._1+x._2)
    }
    v.map(x=>x)
  }
}
