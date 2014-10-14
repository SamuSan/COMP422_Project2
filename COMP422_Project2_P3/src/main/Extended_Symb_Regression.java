package main;

/*
 * This file is part of JGAP.
 *
 * JGAP offers a dual license model containing the LGPL as well as the MPL.
 *
 * For licensing information please see the file license.txt included with JGAP
 * or have a look at the top of class org.jgap.Chromosome which representatively
 * includes the JGAP license policy applicable for any file delivered with JGAP.
 */

import java.io.InputStreamReader;
import java.util.*;

import org.jgap.*;
import org.jgap.gp.*;
import org.jgap.gp.function.*;
import org.jgap.gp.impl.*;
import org.jgap.gp.terminal.*;

/**
 * Example demonstrating Genetic Programming (GP) capabilities of JGAP. Also
 * demonstrates usage of ADF's.<br>
 * The problem is to find a formula for a given truth table (X/Y-pairs).
 * <p>
 * <ul>
 * <li>The setup of the GP is done in method main and specifically in method
 * create()</li>
 * <li>The problem solving process is started via gp.evolve(800) in the main
 * method, with 800 the maximum number of evolutions to take place.
 * <li>The evaluation of the evolved formula is done in fitness function
 * FormulaFitnessFunction, which is implemented in this class, MathProblem
 * </ul>
 * <br>
 * For details, please see the mentioned methods and the fitness function.
 * <p>
 * 
 * @author Klaus Meffert
 * @since 3.0
 */
public class Extended_Symb_Regression extends GPProblem {
	/** String containing the CVS revision. Read out via reflection! */
	private final static String CVS_REVISION = "$Revision: 1.25 $";

	public static Variable vx;

	protected static Float[] x;

	protected static float[] y; 

	public Extended_Symb_Regression(GPConfiguration a_conf)
			throws InvalidConfigurationException {
		super(a_conf);
	}

	/**
	 * This method is used for setting up the commands and terminals that can be
	 * used to solve the problem. In this example an ADF (an automatically
	 * defined function) is used for demonstration purpuses. Using an ADF is
	 * optional. If you want to use one, care about the places marked with
	 * "ADF-relevant:" below. If you do not want to use an ADF, please remove
	 * the below places (and reduce the outer size of the arrays "types",
	 * "argTypes" and "nodeSets" to one). Please notice, that the variables
	 * types, argTypes and nodeSets correspond to each other: they have the same
	 * number of elements and the element at the i'th index of each variable
	 * corresponds to the i'th index of the other variables!
	 * 
	 * @return GPGenotype
	 * @throws InvalidConfigurationException
	 */
	public GPGenotype create() throws InvalidConfigurationException {
		GPConfiguration conf = getGPConfiguration();
		Class[] types = {
		CommandGene.FloatClass };
		Class[][] argTypes = {
		{},
		};
		CommandGene[][] nodeSets = { {
				// We use a variable that can be set in the fitness function.
				// ----------------------------------------------------------
				vx = Variable.create(conf, "X", CommandGene.FloatClass),
				new Multiply(conf, CommandGene.FloatClass),
				new Divide(conf, CommandGene.FloatClass),
				new Subtract(conf, CommandGene.FloatClass),
				new Add(conf, CommandGene.FloatClass),
				new Pow(conf, CommandGene.FloatClass),
                new Sine(conf, CommandGene.FloatClass),
                new If(conf, CommandGene.FloatClass),
                new IfElse(conf, CommandGene.FloatClass),
                new GreaterThan(conf, CommandGene.FloatClass),
                new LesserThan(conf, CommandGene.FloatClass),
                new Or(conf),
                new Equals(conf, CommandGene.FloatClass),
                
				new Terminal(conf, CommandGene.FloatClass, -100d, 100d, false),
		}
		};
		// ------------------------------------------------------------------------
		return GPGenotype.randomInitialGenotype(conf, types, argTypes,
				nodeSets, 20, true);
	}

	/**
	 * Starts the example.
	 * 
	 * @param args
	 *            ignored
	 * @throws Exception
	 * 
	 * @author Klaus Meffert
	 * @since 3.0
	 */
	public static void main(String[] args) throws Exception {
		x = new Float[800];
		y = new float[800];
		String file = args[0];
		Scanner scan = new Scanner(new InputStreamReader(
				ClassLoader.getSystemResourceAsStream(file)));
		int pos=0;
		while(scan.hasNextLine()){
			x[pos] = scan.nextFloat();
			y[pos] = scan.nextFloat();
			pos++;
		}
		System.out.println("X");
		for (Float f : x) {
			System.out.println(f);
		}
		System.out.println("Y");
		for (Float f : y) {
			System.out.println(f);
		}
		// Setup the algorithm's parameters.
		// ---------------------------------
		GPConfiguration config = new GPConfiguration();
		// We use a delta fitness evaluator because we compute a defect rate,
		// not
		// a point score!
		// ----------------------------------------------------------------------
		config.setGPFitnessEvaluator(new DeltaGPFitnessEvaluator());
		config.setMaxInitDepth(4);
		config.setPopulationSize(2000);
		config.setMaxCrossoverDepth(8);
		config.setFitnessFunction(new Extended_Symb_Regression.FormulaFitnessFunction());
		config.setStrictProgramCreation(true);
		config.setCrossoverProb(0.50f);
		config.setMutationProb(0.25f);
		config.setReproductionProb(0.25f);
		GPProblem problem = new Extended_Symb_Regression(config);
		// Create the genotype of the problem, i.e., define the GP commands and
		// terminals that can be used, and constrain the structure of the GP
		// program.
		// --------------------------------------------------------------------
		GPGenotype gp = problem.create();
		gp.setVerboseOutput(true);
		// Start the computation with maximum 800 evolutions.
		// if a satisfying result is found (fitness value almost 0), JGAP stops
		// earlier automatically.
		// --------------------------------------------------------------------
		gp.evolve(2000);
		// Print the best solution so far to the console.
		// ----------------------------------------------
		gp.outputSolution(gp.getAllTimeBest());
		// Create a graphical tree of the best solution's program and write it
		// to
		// a PNG file.
		// ----------------------------------------------------------------------
		problem.showTree(gp.getAllTimeBest(), "mathproblem_best.png");
	}

	/**
	 * Fitness function for evaluating the produced fomulas, represented as GP
	 * programs. The fitness is computed by calculating the result (Y) of the
	 * function/formula for integer inputs 0 to 20 (X). The sum of the
	 * differences between expected Y and actual Y is the fitness, the lower the
	 * better (as it is a defect rate here).
	 */
	public static class FormulaFitnessFunction extends GPFitnessFunction {
		protected double evaluate(final IGPProgram a_subject) {
			return computeRawFitness(a_subject);
		}

		public double computeRawFitness(final IGPProgram ind) {
			double error = 0.0f;
			Object[] noargs = new Object[0];
			// Evaluate function for input numbers 0 to 20.
			// --------------------------------------------
			for (int i = 0; i < 20; i++) {
				// Provide the variable X with the input number.
				// See method create(), declaration of "nodeSets" for where X is
				// defined.
				// -------------------------------------------------------------
				vx.set(x[i]);
				try {
					// Execute the GP program representing the function to be
					// evolved.
					// As in method create(), the return type is declared as
					// float (see
					// declaration of array "types").
					// ----------------------------------------------------------------
					double result = ind.execute_float(0, noargs);
					// Sum up the error between actual and expected result to
					// get a defect
					// rate.
					// -------------------------------------------------------------------
					error +=Math.abs(result - y[i]);
					// If the error is too high, stop evlauation and return
					// worst error
					// possible.
					// ----------------------------------------------------------------
					if (Double.isInfinite(error)) {
						return Double.MAX_VALUE;
					}
				} catch (ArithmeticException ex) {
					// This should not happen, some illegal operation was
					// executed.
					// ------------------------------------------------------------
					System.out.println("x = " + x[i].floatValue());
					System.out.println(ind);
					throw ex;
				}
			}
			// In case the error is small enough, consider it perfect.
			// -------------------------------------------------------
			if (error < 0.001) {
				error = 0.0d;
			}
			return error;
		}
	}
}
