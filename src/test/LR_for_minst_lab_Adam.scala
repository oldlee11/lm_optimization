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
    //test_model.main_deal()//0.91  最后的信息嫡是0.4  在第8次迭代时已经变为0.8

    val updater2=new AdamUpdater()
    updater2.config.setLearningRateByParam(0.1)
    updater2.config.setisMiniBatch(true)
    updater2.config.setminiBatchSize(10000)
    updater2.config.setLearningRatePolicy("Exponential")
    updater2.config.setLrPolicyDecayRate(1)
    updater2.config_Adam.setbeta1(0.9)
    updater2.config_Adam.setbeta2(0.99)
    updater2.config.setL2ByParam(0.05)
    val test_model2=new LR_for_minst_lab(n_epochs,updater2)
    test_model2.main_deal()//如果L2设置为0.1则信息嫡在0.9处反弹。
                           // 如果L2设置为0.05则 正确率= 0.8905，信息嫡在0.76处反弹,并在0.80左右浮动，最后为0.801
  }
}
