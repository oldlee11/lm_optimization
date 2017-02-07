package lm.test

import lm.AdamUpdater

/**
  * Created by liming on 2017/2/4.
  */
object LR_for_minst_lab_Adam {
  def main(args: Array[String]) {
    val n_epochs=100
    val updater=new AdamUpdater()
    updater.config.setLearningRateByParam(0.1)
    updater.config.setisMiniBatch(true)
    updater.config.setminiBatchSize(10000)
    updater.config.setLearningRatePolicy("Exponential")
    updater.config.setLrPolicyDecayRate(1)
    updater.config_Adam.setbeta1(0.9)
    updater.config_Adam.setbeta2(0.99)
    val test_model=new LR_for_minst_lab(n_epochs,updater)
    test_model.main_deal()//0.91  信息嫡是0.4xxx  在第8次迭代时已经变为0.8
  }
}
