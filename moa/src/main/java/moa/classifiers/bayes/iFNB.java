/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package moa.classifiers.bayes;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.ArrayList;


public class iFNB extends NaiveBayes {

    private static long N = 0;
    private static Instance lastStorage = null;
    private static ArrayList<SummaryStatistics> listStatistics = new ArrayList<SummaryStatistics>();

    public static double[] doNaiveBayesPrediction(Instance inst, DoubleVector observedClassDistribution, AutoExpandVector<AttributeClassObserver> attributeObservers) {
        if (lastStorage == null) {
            for (int i = 0; i < inst.numAttributes(); i++) {
                SummaryStatistics stats = new SummaryStatistics();
                listStatistics.add(stats);
                lastStorage = inst;
            }
        }

        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            N = 0;
            votes[classIndex] = observedClassDistribution.getValue(classIndex) / observedClassSum;
            double prod = 1;
            double sum = 0;
            double prodi = 1;
            for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex, inst);
                AttributeClassObserver obs = attributeObservers.get(attIndex);
                if ((obs != null) && (!inst.isMissing(instAttIndex))) {
                    SummaryStatistics get = listStatistics.get(instAttIndex);
                    get.addValue(inst.value(instAttIndex));
                    listStatistics.set(attIndex, get);
                    prod *= listStatistics.get(instAttIndex).getStandardDeviation();
                    double result = (inst.value(instAttIndex) - lastStorage.value(instAttIndex)) / listStatistics.get(instAttIndex).getStandardDeviation();
                    double result1 = (inst.value(instAttIndex) - listStatistics.get(instAttIndex).getMean()) / listStatistics.get(instAttIndex).getStandardDeviation();
                    prodi *= (obs.probabilityOfAttributeValueGivenClass(result, classIndex) + obs.probabilityOfAttributeValueGivenClass(result1, classIndex));
                    N = listStatistics.get(instAttIndex).getN();
                }
                sum += (votes[classIndex] * prodi);
            }
            votes[classIndex] += (sum / (prod * (N - 1)));
        }
        lastStorage = (inst);
        return votes;
    }

    @Override
    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return iFNB.doNaiveBayesPrediction(inst, this.observedClassDistribution,
                this.attributeObservers);
    }
}
