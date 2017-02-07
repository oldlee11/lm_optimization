package lm

/**
  * Created by liming on 2017/2/4.
  */
class AdamUpdater extends BaseUpdater{
  val config_Adam:AdamUpdater_config=new AdamUpdater_config()
  var mPrev:Array[Double]=Array()
  var mPrev_tmp:Array[Double]=Array()
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
        mPrev=mPrev_tmp.map(x=>x/config.getminiBatchSize())
        gPrev=gPrev_tmp.map(x=>x/config.getminiBatchSize())
        mPrev_tmp=Array()
        gPrev_tmp=Array()
        true
      }
    }catch {
      case ex: Exception => false
    }
  }
  override def getGradient(gradient:Array[Double], iteration:Int):Array[Double]={
    /* t是第几次迭代
     * m=beta1*mPrev+(1-beta1)*gradient
     * g=beta2*gPrev+(1-beta2)*gradient^2
     * m2=m/(1-beta1^(t+1))
     * g2=g/(1-beta2^(t+1))
     * 参数变化量Param_Diff=m2*LearningRateByParam/(g2的平方+epsilon)
     *
     * * 注意在批量模式下 mPrev 是一批样本的m的均值,在getGradient函数内做累加,在update函数内做平均
     *                    gPrev 是一批样本的g的均值,在getGradient函数内做累加,在update函数内做平均
     * */
    val (m,g)=if(mPrev.length==0){
      (gradient.map(x=>(1-config_Adam.getbeta1())*x),
      gradient.map(x=>(1-config_Adam.getbeta2())*x*x))
    }else{
      (mPrev.
        zip(gradient).
        map(x=>config_Adam.getbeta1()*x._1+(1-config_Adam.getbeta1())*x._2),
      gPrev.
        zip(gradient).
        map(x=>config_Adam.getbeta2()*x._1+(1-config_Adam.getbeta2())*x._2*x._2))
    }
    val m2=m.map(x=>x/(1-math.pow(config_Adam.getbeta1(),iteration+1)))
    val g2=g.map(x=>x/(1-math.pow(config_Adam.getbeta2(),iteration+1)))
    //更新m/gprev(用于批量模式)
    if(mPrev_tmp.length==0){
      mPrev_tmp=m
      gPrev_tmp=g
    }else{
      mPrev_tmp=mPrev_tmp.zip(m).map(x=>x._1+x._2)
      gPrev_tmp=gPrev_tmp.zip(g).map(x=>x._1+x._2)
    }
    m2.
      zip(g2).
      map(x=>x._1*config.getLearningRateByParam()/(math.sqrt(x._2)+1e-8))
  }
}
