package lm

/**
  * Created by liming on 2017/2/3.
  * 最简单的随机梯度下降法
  */
class SgdUpdater extends BaseUpdater{
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
        true
      }
    }catch {
      case ex: Exception => false
    }
  }
  override def getGradient(gradient:Array[Double], iteration:Int):Array[Double]={
    gradient.map(x=>x*config.getLearningRateByParam())
  }
}
