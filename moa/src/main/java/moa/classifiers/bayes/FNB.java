package moa.classifiers.bayes;


import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;

public class FNB extends NaiveBayes {
    private static Instances localStorage = null;
    private static ArrayList<DescriptiveStatistics> listStatistics = new ArrayList<DescriptiveStatistics>();

    public static double[] doNaiveBayesPrediction(Instance inst, DoubleVector observedClassDistribution, AutoExpandVector<AttributeClassObserver> attributeObservers) {
        if (localStorage == null) {
            for (int i = 0; i < inst.numAttributes(); i++) {
                DescriptiveStatistics stats = new DescriptiveStatistics();
                stats.setWindowSize(1000);
                listStatistics.add(stats);
                localStorage = new Instances(inst.dataset(), 1000);
            }
        }

        localStorage.add(inst);
        double[] votes = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();
        for (int classIndex = 0; classIndex < votes.length; classIndex++) {
            votes[classIndex] = observedClassDistribution.getValue(classIndex) / observedClassSum;
            double prod = 1;
            double sum = 0;
            for (int rows = 0; rows < localStorage.size(); rows++) {
                double prodi = 1;
                for (int attIndex = 0; attIndex < inst.numAttributes() - 1; attIndex++) {

                    int instAttIndex = modelAttIndexToInstanceAttIndex(attIndex, inst);
                    AttributeClassObserver obs = attributeObservers.get(attIndex);
                    if ((obs != null) && (!inst.isMissing(instAttIndex))) {
                        DescriptiveStatistics get = listStatistics.get(attIndex);
                        get.addValue(inst.value(instAttIndex));
                        listStatistics.set(attIndex, get);
                        prod *= listStatistics.get(instAttIndex).getStandardDeviation();
                        double result = (inst.value(instAttIndex) - localStorage.get(rows).value(instAttIndex)) / listStatistics.get(instAttIndex).getStandardDeviation();

                        prodi *= obs.probabilityOfAttributeValueGivenClass(result, classIndex);

                    }

                }
                sum += prodi;
            }
            votes[classIndex] += (sum / (prod * localStorage.size()));
        }
        if (localStorage.size() >= 1000) {
            localStorage.delete(0);

        }
        localStorage.add(inst);
        return votes;
    }

    @Override
    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }
}
