package lm.test

import lm.NesterovsUpdater

/**
  * Created by liming on 2017/2/4.
  */
object LR_for_minst_lab_Nesterovs {
  def main(args: Array[String]) {
    val n_epochs=100
    val updater=new NesterovsUpdater()
    updater.config.setLearningRateByParam(0.1)
    updater.config.setisMiniBatch(true)
    updater.config.setminiBatchSize(10000)
    updater.config.setLearningRatePolicy("Exponential")
    updater.config.setLrPolicyDecayRate(1)
    updater.config_Nesterovs.setMomentumSchedule(0.8)
    val test_model=new LR_for_minst_lab(n_epochs,updater)
    test_model.main_deal()//0.84

  }
}
