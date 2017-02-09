package lm

import java.io.Serializable

/**
  * refer to deeplearning4j
  * Created by liming on 2017/2/3.
  * 所有梯度下降的父类
  * 不同类型的梯度下降子类需要重构getGradient和update函数
  * 顶层函数为update(更新参数)
  *
  * 如果是map-reduce思想的话
  * gen_ParamDiff用于map过程,对每个样本生成一个系数param变化的向量
  * update是用于reduce过程，对一批样本产生的系数变化向量做平均，并更新最后的系数
  */
class BaseUpdater extends Serializable{

  val config:BaseUpdater_config=new BaseUpdater_config()


  /** 批量模式下(非批量模式下则仅有一组样本数据输入)
    * 一次迭代中,更新系数paramVal的过程
    * param paramVal              模型的系数(最终要优化的参数),最后会修改paramVal的数据
    * param gradient_org_batch    一批样本经过梯度下降后产生的梯度
    * param iteration             本次迭代属于第几次迭代
    *
    * 需要重构实现
    */
  def update(paramVal:Array[Double],
             gradient_org_batch:Array[Array[Double]],
             iteration:Int):Boolean ={
    false
  }


  /** 一次迭代中,一个样本经过模型梯度下降后,给出的系数的变化量
    * paramVal              模型的系数(最终要优化的参数)
    * param gradient_org    一个样本内产生的梯度(根据不同的模型以及模型内使用的LOSS损失函数来定)
    * param iteration       本次迭代属于第几次迭代
    */
  def gen_ParamDiff(paramVal:Array[Double],
                     gradient_org:Array[Double],
                     iteration:Int):Array[Double]={
    try{
      val paramVallen:Int=paramVal.length
      //系数和梯度必须要一一匹配(长度相等)
      if(gradient_org.length!=paramVallen){
        Array()
      }else{
        //更新学习率
        applyLrDecayPolicy(iteration)
        //根据不同的梯度方法来 计算梯度gradient2
        val gradient2:Array[Double] = getGradient(gradient=gradient_org,iteration=iteration);
        if(gradient2.length==0){
          Array()
        }else{
          //L1和l2型正则化处理,如果是批量模式则除以minbachsize,更新梯度gradient2
          postApply(gradient=gradient2,paramVal=paramVal);
          gradient2
        }
      }
    }catch {
      case ex: Exception => Array()
    }
  }



  //每次迭代更新学习率
  def applyLrDecayPolicy(iteration:Int):Unit={
    val lr:Double=config.getLearningRateByParam()           //学习率
    val lrdecayPolicy:String=config.getLearningRatePolicy() //学习率更新策略
    val lrdecayRate:Double=config.getLrPolicyDecayRate()    //学习率更新策略下使用的参数
    //根据更新策略参数lrdecayPolicy和其它参数对学习率lr进行更新
    lrdecayPolicy match {
      /*
      lr(t)=lr * lrdecayRate^t
      即 第t次迭代的学习率=初始化的lr * lrdecayRate的t次幂
      即 lr(t+1)=lr(t)*lrdecayRate
      * */
      case "Exponential" => config.setLearningRateByParam(lr * Math.pow(lrdecayRate, iteration))
      /*
      * lr(t)=lr / ((1+lrdecayRate*t)^LrPolicyPower)
      * 即 第t次迭代的学习率=初始化的lr 除以 1+lrdecayRate*t的LrPolicyPower次幂
      * */
      case "Inverse" => config.setLearningRateByParam(lr / Math.pow((1 + lrdecayRate * iteration), config.getLrPolicyPower()))
      /*
      * lr(t)=lr * lrdecayRate^(t/LrPolicySteps)
      * 即 每经过LrPolicySteps次迭代,lr乘以一次lrdecayRate
      * */
      case "Step" => config.setLearningRateByParam(lr * Math.pow(lrdecayRate, Math.floor(iteration / config.getLrPolicySteps())))
      case "Sigmoid" => config.setLearningRateByParam(lr / (1 + Math.exp(-lrdecayRate * (iteration - config.getLrPolicySteps()))))
      //什么也不做
      case "NoChange" =>
    }
  }



  //根据不同的梯度下降算法来重构该函数
  def getGradient(gradient:Array[Double], iteration:Int):Array[Double]={
    null
  }



  //L1和l2型正则化处理,如果是批量模式则除以minbachsize
  def postApply(gradient:Array[Double],paramVal:Array[Double]):Unit={
    val paramVallen:Int=paramVal.length
    //(gradient+paramVal*L2Param+L1Param)
    if (!config.isNoneL2ByParam()) {
      //如果设置了L2超参数,则更新gradient
      (0 until paramVallen).map(index => gradient(index) = gradient(index) + paramVal(index) * config.getL2ByParam())
    }
    if (!config.isNoneL1ByParam()) {
      //如果设置了L1超参数,则更新gradient
      (0 until paramVallen).map(index => gradient(index) = gradient(index) + sign(paramVal(index))*config.getL1ByParam())
    }
    if (config.getisMiniBatch()) {
      //  在批量模式下,  =梯度/批量样本的个数miniBatchSize
      (0 until paramVallen).map(index => gradient(index) = gradient(index) / config.getminiBatchSize())
    }
  }

  //取input的符号 +1 好是 -1 以及 0
  def sign(input:Double):Double=if(input>=1e-6) 1.0 else if(input<= -1e-6) -1.0 else 0.0
}
