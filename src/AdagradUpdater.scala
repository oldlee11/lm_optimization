package lm

/**
  * Created by liming on 2017/2/4.
  */
class AdagradUpdater extends BaseUpdater{
  var gPrev:Array[Double]=Array()
  var gPrev_tmp:Array[Double]=Array()

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
        gPrev=gPrev_tmp.map(x=>x/config.getminiBatchSize())
        gPrev_tmp=Array()
        true
      }
    }catch {
      case ex: Exception => false
    }
  }
  override def getGradient(gradient:Array[Double], iteration:Int):Array[Double]={
    /*
    * g=gPrev+gradient^2
    * 参数变化量Param_Diff=gradient*LearningRateByParam/(g的平方+epsilon)
    * 注意在批量模式下 gPrev 是一批样本的g的均值,在getGradient函数内做累加,在update函数内做平均
    * */
    val g:Array[Double]=if(gPrev.length==0){
      gradient.map(x=>x*x*config.getLearningRateByParam())
    }else{
      gPrev.
        zip(gradient.map(x=>x*x*config.getLearningRateByParam())).
        map(x=>x._1+x._2)
    }
    //更新gprev(用于批量模式)
    if(gPrev_tmp.length==0){
      gPrev_tmp=g
    }else{
      gPrev_tmp=gPrev_tmp.zip(g).map(x=>x._1+x._2)
    }
    gradient.
      zip(g).
      map(x=>x._1*config.getLearningRateByParam()/(math.sqrt(x._2)+1e-8))

  }
}
