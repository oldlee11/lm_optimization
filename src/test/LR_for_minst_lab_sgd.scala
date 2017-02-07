package lm.test

import lm.SgdUpdater

/**
  * Created by liming on 2017/2/4.
  */
object LR_for_minst_lab_sgd {
  def main(args: Array[String]) {
    val n_epochs=100
    val updater=new SgdUpdater()
    updater.config.setLearningRateByParam(0.9)
    updater.config.setisMiniBatch(true)
    updater.config.setminiBatchSize(10000)
    updater.config.setLearningRatePolicy("Exponential")
    updater.config.setLrPolicyDecayRate(1)
    val test_model=new LR_for_minst_lab(n_epochs,updater)
    //test_model.main_deal()

    val updater2=new SgdUpdater()
    updater2.config.setLearningRateByParam(0.1)
    updater2.config.setisMiniBatch(true)
    updater2.config.setminiBatchSize(10000)
    updater2.config.setLearningRatePolicy("Exponential")
    updater2.config.setLrPolicyDecayRate(1)
    updater2.config.setL2ByParam(0.01)
    val test_model2=new LR_for_minst_lab(n_epochs,updater2)
    test_model2.main_deal()//0.8518,信息嫡=1.25(100次迭代太小了,可以增加迭代次数或者是学习率,学习率=0.9也可以)

    val updater3=new SgdUpdater()
    updater3.config.setLearningRateByParam(0.95)
    updater3.config.setisMiniBatch(true)
    updater3.config.setminiBatchSize(10000)
    updater3.config.setLearningRatePolicy("Exponential")
    updater3.config.setLrPolicyDecayRate(1)
    updater3.config.setL1ByParam(0.01)
    val test_model3=new LR_for_minst_lab(n_epochs,updater3)
    //test_model3.main_deal()
  }
}
