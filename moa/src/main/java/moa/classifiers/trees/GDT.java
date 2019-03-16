/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package moa.classifiers.trees;

import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.bayes.iFNB;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.splitcriteria.SplitCriterion;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class GDT extends HoeffdingTree {
    private static int insttoleafdepth;
    private static int decisionNodeCounti;

    public GDT() {
        super();
        this.leafpredictionOption = new MultiChoiceOption("leafprediction", 'l', "Leaf prediction to use.", new String[]{"MC", "NB", "NBAdaptive", "FNB", "iFNB"}, new String[]{"Majority class", "Naive Bayes", "Naive Bayes Adaptive", "Flexible Naive Bayes", "iFNB"}, 2);
        decisionNodeCounti = this.decisionNodeCount;
    }

    private static double getC(double threshold) {
        return (1 - threshold) / threshold;
    }

    private static double getQ() {
        double e = 2.71828;
        double t1, t2, t3, t4;
        double C = getC(0.95);

        t1 = ((Math.log(e * e)) * (Math.log(e * e))) / (C * e * e);
        t2 = ((Math.log(e * e)) * (Math.log(e * e))) / (e * e);
        t3 = (Math.log(e)) / e;
        t4 = ((Math.log(2 * C)) * (Math.log(2 * C))) / 4;

        return t1 + t2 + t3 + t4;
    }

    private static double getz(double confidence) {// calculate z using confidence
        double z = 1.64854;
        return z;
    }

    public static double computeGDTBound(double range, double confidence,
                                         double n) {
        double z = getz(confidence);
        double Q = getQ();
        return z * (Math.sqrt(2 * Q) / Math.sqrt(n));
    }

    @Override
    public String getPurposeString() {
        return "KDE classifier: performs Naive bayesian prediction while computing probability density estimation.";
    }

    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations) {
        LearningNode ret;
        int predictionOption = this.leafpredictionOption.getChosenIndex();
        if (predictionOption == 0) { //MC
            ret = new ActiveLearningNode(initialClassObservations);
        } else if (predictionOption == 1) { //NB
            ret = new LearningNodeNB(initialClassObservations);
        } else if (predictionOption == 2) { //NBAdaptive
            ret = new LearningNodeNBAdaptive(initialClassObservations);
        } else {
            ret = new LearningNodeIFNB(initialClassObservations);
        }
        return ret;
    }

    @Override
    protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,
                                  int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeGDTBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) {
                    shouldSplit = true;
                }
                // }
                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2 - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }
            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);
                } else {
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(), splitDecision.numSplits());
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }
                // manage memory
                enforceTrackerLimit();
            }
        }
    }

    public static class LearningNodeIFNB
            extends LearningNodeNB {
        private static final long serialVersionUID = 1L;

        public LearningNodeIFNB(double[] initialClassObservations) {
            super(initialClassObservations);

        }

        @Override
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            GDT.insttoleafdepth++;
            return iFNB.doNaiveBayesPrediction(inst, this.observedClassDistribution, this.attributeObservers);
        }

        @Override
        public void disableAttribute(int attIndex) {
        }
    }
}
