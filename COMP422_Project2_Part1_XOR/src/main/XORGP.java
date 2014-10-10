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

import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import javax.swing.tree.TreeNode;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.SimpleLayout;
import org.jgap.InvalidConfigurationException;
import org.jgap.gp.CommandGene;
import org.jgap.gp.GPFitnessFunction;
import org.jgap.gp.IGPProgram;
import org.jgap.gp.function.ADF;
import org.jgap.gp.function.Abs;
import org.jgap.gp.function.Add;
import org.jgap.gp.function.Add3;
import org.jgap.gp.function.Add4;
import org.jgap.gp.function.And;
import org.jgap.gp.function.ArcCosine;
import org.jgap.gp.function.ArcSine;
import org.jgap.gp.function.ArcTangent;
import org.jgap.gp.function.Ceil;
import org.jgap.gp.function.Cosine;
import org.jgap.gp.function.Divide;
import org.jgap.gp.function.Equals;
import org.jgap.gp.function.Exp;
import org.jgap.gp.function.Floor;
import org.jgap.gp.function.ForLoop;
import org.jgap.gp.function.ForXLoop;
import org.jgap.gp.function.GreaterThan;
import org.jgap.gp.function.If;
import org.jgap.gp.function.IfDyn;
import org.jgap.gp.function.IfElse;
import org.jgap.gp.function.Increment;
import org.jgap.gp.function.LesserThan;
import org.jgap.gp.function.Log;
import org.jgap.gp.function.Loop;
import org.jgap.gp.function.Max;
import org.jgap.gp.function.Min;
import org.jgap.gp.function.Modulo;
import org.jgap.gp.function.Multiply;
import org.jgap.gp.function.Multiply3;
import org.jgap.gp.function.Not;
import org.jgap.gp.function.Or;
import org.jgap.gp.function.Pop;
import org.jgap.gp.function.Pow;
import org.jgap.gp.function.Push;
import org.jgap.gp.function.Round;
import org.jgap.gp.function.Sine;
import org.jgap.gp.function.StoreTerminal;
import org.jgap.gp.function.SubProgram;
import org.jgap.gp.function.Subtract;
import org.jgap.gp.function.Tangent;
import org.jgap.gp.function.Tupel;
import org.jgap.gp.function.Xor;
import org.jgap.gp.impl.DeltaGPFitnessEvaluator;
import org.jgap.gp.impl.GPConfiguration;
import org.jgap.gp.impl.GPGenotype;
import org.jgap.gp.impl.GPPopulation;
import org.jgap.gp.impl.JGAPTreeBranchRenderer;
import org.jgap.gp.impl.JGAPTreeNode;
import org.jgap.gp.impl.JGAPTreeNodeRenderer;
import org.jgap.gp.impl.ProgramChromosome;
import org.jgap.gp.impl.TournamentSelector;
import org.jgap.gp.terminal.Constant;
import org.jgap.gp.terminal.Terminal;
import org.jgap.gp.terminal.Variable;
import org.jgap.util.NumberKit;
import org.jgap.util.SystemKit;
import org.jgap.util.tree.TreeBranchRenderer;
import org.jgap.util.tree.TreeNodeRenderer;
import org.jgap.util.tree.TreeVisualizer;

import examples.gp.symbolicRegression.AndD;
import examples.gp.symbolicRegression.Gamma;
import examples.gp.symbolicRegression.Gaussian;
import examples.gp.symbolicRegression.Hill;
import examples.gp.symbolicRegression.Logistic;
import examples.gp.symbolicRegression.ModuloD;
import examples.gp.symbolicRegression.NotD;
import examples.gp.symbolicRegression.OrD;
import examples.gp.symbolicRegression.Sigmoid;
import examples.gp.symbolicRegression.Sign;
import examples.gp.symbolicRegression.Sqrt;
import examples.gp.symbolicRegression.Step;
import examples.gp.symbolicRegression.XorD;

/*
 *
 * Symbolic Regression in JGAP.
 *
 * This program is based on the JGAP example MathProblem.java with
 * some generality.
 *
 * TODO:
 *  - option for ignoring specific variables
 *  - option for stopping:
 *     - running forever
 *     - after a specific time,
 *     - when a specific fitness value is reached
 *  - calculate the number of data rows automatically (i.e. skip num_row)
 *  - show similiar solutions (with the same fitness as the best)
 *  - check if there are any more parameters in GPConfiguration to handle.
 *  - accept nominal values in the data section; then converted to
 *    numeric values.
 *  - add fitness metrics.
 *  - punish longer solutions
 *  - support for different "main" return classes, i.e. not just DoubleClass
 *  - correlation coefficient, and other statistical measures, e.g.
 *    R-squared, mean squared error, mean absolut error, minimum error,
 *    maximum error
 *  - more/better error checks
 *  - more building blocks, a la Eureqa http://ccsl.mae.cornell.edu/eureqa_ops
 *  - support for derivata (a la Eureqa)? This may be hard...
 *  - integrate with Weka?
 *  - simplify the best solution with a CAS?
 *
 * This program was written by Hakan Kjellerstrand (hakank@bonetmail.com)
 * Also, see my JGAP page http://www.hakank.org/jgap/
 *
 * @since 3.5
 */
public class XORGP extends GPProblem {
	private transient static Logger LOGGER = Logger
			.getLogger(XORGP.class);

	/*
	 * public variables which may be changed by configuration file
	 */

	// number of variables to use (output variable is excluded)
	public static int numInputVariables;

	// the variables to use (of size numInputVariables)
	public static Variable[] variables;

	// variable name
	public static String[] variableNames;

	// index of the output variable
	public static Integer outputVariable; // default last

	public static int[] ignoreVariables; // TODO

	// constants
	public static ArrayList<Double> constants = new ArrayList<Double>();

	// size of data
	public static int numRows;

	// the data (as Double)
	// Note: the last row is the output variable per default
	protected static Double[][] data;

	// If we have found a perfect solution.
	public static boolean foundPerfect = false;

	// standard GP parameters
	public static int minInitDepth = 2;

	public static int maxInitDepth = 4;

	public static int populationSize = 100;

	public static int maxCrossoverDepth = 8;

	public static int programCreationMaxTries = 5;

	public static int numEvolutions = 1800;

	public static boolean verboseOutput = true;

	public static int maxNodes = 21;

	public static double functionProb = 0.9d;

	public static float reproductionProb = 0.1f; // float

	public static float mutationProb = 0.1f; // float

	public static double crossoverProb = 0.9d;

	public static float dynamizeArityProb = 0.08f; // float

	public static double newChromsPercent = 0.3d;

	public static int tournamentSelectorSize = 0;

	// lower/upper ranges for the Terminal
	public static double lowerRange = -10.0d;

	public static double upperRange = -10.0d;

	// Should the terminal be a wholenumber or not?
	public static boolean terminalWholeNumbers = true;

	public static String returnType = "DoubleClass"; // not used yet

	public static String presentation = "";

	// Using ADF
	public static int adfArity = -1;

	public static String adfType = "double";

	public static boolean useADF = false;

	// list of functions (as strings)
	public static String[] functions = { "Multiply", "Divide", "Add",
			"Subtract" };

	// list of functions for ADF
	public static String[] adfFunctions = { "Multiply3", "Divide", "Add3",
			"Subtract" };

	// Should we punish length of solutions?
	// Note: Very simplistic version.
	// public static boolean punishLength = false;

	// if the values are too small we may want to scale
	// the error
	public static double scaleError = -1.0d;

	// "bumping" is when we found a "perfect solution" and
	// want to see more "perfect solutions"
	public static boolean bumpPerfect = true;

	// the limit for which we should show all (different) solutions
	public static Double bumpValue = 0.00;

	// checks for already shown solution when bumping
	private static HashMap<String, Integer> foundSolutions = new HashMap<String, Integer>();

	// timing
	public static long startTime;

	public static long endTime;

	// if > 0.0d -> stop if the fitness is below or equal
	// this value. TODO!
	public static double stopCriteria = -1.0d;

	public static boolean showPopulation = false;

	public static boolean showSimiliar = false;

	public XORGP(GPConfiguration a_conf)
			throws InvalidConfigurationException {
		super(a_conf);
	}

	/**
	 * This method is used for setting up the commands and terminals that can be
	 * used to solve the problem.
	 * 
	 * @return GPGenotype
	 * @throws InvalidConfigurationException
	 */
	public GPGenotype create() throws InvalidConfigurationException {
		GPConfiguration conf = getGPConfiguration();
		// At first, we define the return type of the GP program.
		// ------------------------------------------------------
		// Then, we define the arguments of the GP parts. Normally, only for
		// ADF's
		// there is a specification here, otherwise it is empty as in first
		// case.
		// -----------------------------------------------------------------------
		Class[] types;
		Class[][] argTypes;
		if (useADF) {
			if ("boolean".equals(adfType)) {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.BooleanClass };
			} else if ("integer".equals(adfType)) {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.IntegerClass };
			} else {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.DoubleClass };
			}
			Class[] adfs = new Class[adfArity];
			for (int i = 0; i < adfArity; i++) {
				if ("boolean".equals(adfType)) {
					adfs[i] = CommandGene.BooleanClass;
				} else if ("integer".equals(adfType)) {
					adfs[i] = CommandGene.IntegerClass;
				} else {
					adfs[i] = CommandGene.DoubleClass;
				}
			}
			argTypes = new Class[][] { {}, adfs };
		} else {
			types = new Class[] { CommandGene.DoubleClass };
			argTypes = new Class[][] { {} };
		}
		if (useADF) {
		} else {
		}
		// Next, we define the set of available GP commands and terminals to
		// use.
		// Please see package org.jgap.gp.function and org.jgap.gp.terminal
		// You can easily add commands and terminals of your own.
		// ----------------------------------------------------------------------
		CommandGene[] commands = makeCommands(conf, functions, lowerRange,
				upperRange, "plain");
		// Create the node sets
		int command_len = commands.length;
		CommandGene[][] nodeSets = new CommandGene[1][numInputVariables
				+ command_len];
		// the variables:
		// 1) in the nodeSets matrix
		// 2) as variables (to be used for fitness checking)
		// --------------------------------------------------
		variables = new Variable[numInputVariables];
		int variableIndex = 0;
		for (int i = 0; i < numInputVariables + 1; i++) {
			String variableName = variableNames[i];
			if (i != outputVariable) {
				if (variableNames != null && variableNames.length > 0) {
					variableName = variableNames[i];
				}
				variables[variableIndex] = Variable.create(conf, variableName,
						CommandGene.DoubleClass);
				nodeSets[0][variableIndex] = variables[variableIndex];
				System.out.println("input variable: "
						+ variables[variableIndex]);
				variableIndex++;
			}
		}
		// assign the functions/terminals
		// ------------------------------
		for (int i = 0; i < command_len; i++) {
			System.out.println("function1: " + commands[i]);
			nodeSets[0][i + numInputVariables] = commands[i];
		}
		// ADF functions in the second array in nodeSets
		if (useADF) {
			CommandGene[] adfCommands = makeCommands(conf, adfFunctions,
					lowerRange, upperRange, "ADF");
			int adfLength = adfCommands.length;
			nodeSets[1] = new CommandGene[adfLength];
			for (int i = 0; i < adfLength; i++) {
				System.out.println("function2: " + adfCommands[i]);
				nodeSets[1][i] = adfCommands[i];
			}
		}
		// this is experimental.
		boolean[] full;
		if (useADF) {
			full = new boolean[] { true, true };
		} else {
			full = new boolean[] { true };
		}
		// Create genotype with initial population. Here, we use the
		// declarations made above:
		// ----------------------------------------------------------
		return GPGenotype.randomInitialGenotype(conf, types, argTypes,
				nodeSets, maxNodes, verboseOutput);
		// this is experimental
		// return GPGenotype.randomInitialGenotype(conf, types, argTypes,
		// nodeSets,
		// minDepths,maxDepths, maxNodes, fullModeAllowed,verboseOutput);

	}

	public static GPGenotype createNewGenoForTest(GPPopulation pop,
			GPConfiguration conf1) throws InvalidConfigurationException {
		GPConfiguration conf = conf1;
		// At first, we define the return type of the GP program.
		// ------------------------------------------------------
		// Then, we define the arguments of the GP parts. Normally, only for
		// ADF's
		// there is a specification here, otherwise it is empty as in first
		// case.
		// -----------------------------------------------------------------------
		Class[] types;
		Class[][] argTypes;
		if (useADF) {
			if ("boolean".equals(adfType)) {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.BooleanClass };
			} else if ("integer".equals(adfType)) {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.IntegerClass };
			} else {
				types = new Class[] { CommandGene.DoubleClass,
						CommandGene.DoubleClass };
			}
			Class[] adfs = new Class[adfArity];
			for (int i = 0; i < adfArity; i++) {
				if ("boolean".equals(adfType)) {
					adfs[i] = CommandGene.BooleanClass;
				} else if ("integer".equals(adfType)) {
					adfs[i] = CommandGene.IntegerClass;
				} else {
					adfs[i] = CommandGene.DoubleClass;
				}
			}
			argTypes = new Class[][] { {}, adfs };
		} else {
			types = new Class[] { CommandGene.DoubleClass };
			argTypes = new Class[][] { {} };
		}
		if (useADF) {
		} else {
		}
		// Next, we define the set of available GP commands and terminals to
		// use.
		// Please see package org.jgap.gp.function and org.jgap.gp.terminal
		// You can easily add commands and terminals of your own.
		// ----------------------------------------------------------------------
		CommandGene[] commands = makeCommands(conf, functions, lowerRange,
				upperRange, "plain");
		// Create the node sets
		int command_len = commands.length;
		CommandGene[][] nodeSets = new CommandGene[1][numInputVariables
				+ command_len];
		// the variables:
		// 1) in the nodeSets matrix
		// 2) as variables (to be used for fitness checking)
		// --------------------------------------------------
		variables = new Variable[numInputVariables];
		int variableIndex = 0;
		for (int i = 0; i < numInputVariables + 1; i++) {
			String variableName = variableNames[i];
			if (i != outputVariable) {
				if (variableNames != null && variableNames.length > 0) {
					variableName = variableNames[i];
				}
				variables[variableIndex] = Variable.create(conf, variableName,
						CommandGene.DoubleClass);
				nodeSets[0][variableIndex] = variables[variableIndex];
				System.out.println("input variable: "
						+ variables[variableIndex]);
				variableIndex++;
			}
		}
		// assign the functions/terminals
		// ------------------------------
		for (int i = 0; i < command_len; i++) {
			System.out.println("function1: " + commands[i]);
			nodeSets[0][i + numInputVariables] = commands[i];
		}
		// ADF functions in the second array in nodeSets
		if (useADF) {
			CommandGene[] adfCommands = makeCommands(conf, adfFunctions,
					lowerRange, upperRange, "ADF");
			int adfLength = adfCommands.length;
			nodeSets[1] = new CommandGene[adfLength];
			for (int i = 0; i < adfLength; i++) {
				System.out.println("function2: " + adfCommands[i]);
				nodeSets[1][i] = adfCommands[i];
			}
		}
		// this is experimental.
		boolean[] full;
		if (useADF) {
			full = new boolean[] { true, true };
		} else {
			full = new boolean[] { true };
		}
		// Create genotype with initial population. Here, we use the
		// declarations made above:
		// ----------------------------------------------------------
//		pop.create(0, types, argTypes, nodeSets, new int [0], new int [0], 0, true, 21, new boolean [0], 0);
		
		GPGenotype gp = new GPGenotype(conf, pop, types, argTypes, nodeSets,
				new int[0], new int[0], maxNodes);
		return gp;
		// this is experimental
		// return GPGenotype.randomInitialGenotype(conf, types, argTypes,
		// nodeSets,
		// minDepths,maxDepths, maxNodes, fullModeAllowed,verboseOutput);

	}

	public static void readFile(String file) {
		try {
			BufferedReader inr = new BufferedReader(new InputStreamReader(
					ClassLoader.getSystemResourceAsStream(file)));
			String str;
			int lineCount = 0;
			boolean gotData = false;
			ArrayList<Double[]> theData = new ArrayList<Double[]>();
			//
			// read the lines
			//
			while ((str = inr.readLine()) != null) {
				lineCount++;
				str = str.trim();
				// ignore empty lines or comments, i.e. lines starting with
				// either # or %
				// ----------------------------------------------------------------------
				if (str.startsWith("#") || str.startsWith("%")
						|| str.length() == 0) {
					continue;
				}
				if ("data".equals(str)) {
					gotData = true;
					continue;
				}
				if (gotData) {
					// Read the data rows
					// ------------------
					String[] dataRowStr = str.split("[\\s,]+");
					int len = dataRowStr.length;
					Double[] dataRow = new Double[len];
					for (int i = 0; i < len; i++) {
						dataRow[i] = Double.parseDouble(dataRowStr[i]);
					}
					theData.add(dataRow);
				} else {
					// Check for parameters on the form
					// parameter: value(s)
					// --------------------------------
					if (str.contains(":")) {
						String row[] = str.split(":\\s*");
						// Now check each parameter
						if ("return_type".equals(row[0])) {
							returnType = row[1];
						} else if ("presentation".equals(row[0])) {
							presentation = row[1];
						} else if ("num_input_variables".equals(row[0])) {
							numInputVariables = Integer.parseInt(row[1]);
						} else if ("num_rows".equals(row[0])) {
							numRows = Integer.parseInt(row[1]);
						} else if ("terminal_range".equals(row[0])) {
							String[] ranges = row[1].split("\\s+");
							lowerRange = Double.parseDouble(ranges[0]);
							upperRange = Double.parseDouble(ranges[1]);
						} else if ("terminal_wholenumbers".equals(row[0])) {
							terminalWholeNumbers = Boolean.parseBoolean(row[1]);
						} else if ("max_init_depth".equals(row[0])) {
							maxInitDepth = Integer.parseInt(row[1]);
						} else if ("min_init_depth".equals(row[0])) {
							minInitDepth = Integer.parseInt(row[1]);
						} else if ("program_creation_max_tries".equals(row[0])) {
							programCreationMaxTries = Integer.parseInt(row[1]);
						} else if ("population_size".equals(row[0])) {
							populationSize = Integer.parseInt(row[1]);
						} else if ("max_crossover_depth".equals(row[0])) {
							maxCrossoverDepth = Integer.parseInt(row[1]);
						} else if ("function_prob".equals(row[0])) {
							functionProb = Double.parseDouble(row[1]);
						} else if ("reproduction_prob".equals(row[0])) {
							reproductionProb = Float.parseFloat(row[1]);
						} else if ("mutation_prob".equals(row[0])) {
							mutationProb = Float.parseFloat(row[1]);
						} else if ("crossover_prob".equals(row[0])) {
							crossoverProb = Double.parseDouble(row[1]);
						} else if ("dynamize_arity_prob".equals(row[0])) {
							dynamizeArityProb = Float.parseFloat(row[1]);
						} else if ("new_chroms_percent".equals(row[0])) {
							newChromsPercent = Double.parseDouble(row[1]);
						} else if ("num_evolutions".equals(row[0])) {
							numEvolutions = Integer.parseInt(row[1]);
						} else if ("max_nodes".equals(row[0])) {
							maxNodes = Integer.parseInt(row[1]);
						} else if ("bump".equals(row[0])) {
							bumpPerfect = Boolean.parseBoolean(row[1]);
						} else if ("bump_value".equals(row[0])) {
							bumpValue = Double.parseDouble(row[1]);
						} else if ("functions".equals(row[0])) {
							functions = row[1].split("[\\s,]+");
						} else if ("adf_functions".equals(row[0])) {
							adfFunctions = row[1].split("[\\s,]+");
						} else if ("variable_names".equals(row[0])) {
							variableNames = row[1].split("[\\s,]+");
						} else if ("output_variable".equals(row[0])) {
							outputVariable = Integer.parseInt(row[1]);
						} else if ("ignore_variables".equals(row[0])) {
							String[] ignoreVariablesS = row[1].split("[\\s,]+");
							ignoreVariables = new int[ignoreVariablesS.length];
							// TODO: make it a HashMap instead!
							for (int i = 0; i < ignoreVariablesS.length; i++) {
								ignoreVariables[i] = Integer
										.parseInt(ignoreVariablesS[i]);
							}
						} else if ("constant".equals(row[0])) {
							Double constant = Double.parseDouble(row[1]);
							constants.add(constant);
						} else if ("adf_arity".equals(row[0])) {
							adfArity = Integer.parseInt(row[1]);
							System.out.println("ADF arity " + adfArity);
							if (adfArity > 0) {
								useADF = true;
							}
						} else if ("adf_type".equals(row[0])) {
							adfType = row[1];
							// } else if ("punish_length".equals(row[0])) {
							// punishLength = Boolean.parseBoolean(row[1]);

						} else if ("tournament_selector_size".equals(row[0])) {
							tournamentSelectorSize = Integer.parseInt(row[1]);
						} else if ("scale_error".equals(row[0])) {
							scaleError = Double.parseDouble(row[1]);
						} else if ("stop_criteria".equals(row[0])) {
							stopCriteria = Double.parseDouble(row[1]);
						} else if ("show_population".equals(row[0])) {
							showPopulation = Boolean.parseBoolean(row[1]);
						} else if ("show_similiar".equals(row[0])) {
							showSimiliar = Boolean.parseBoolean(row[1]);
						} else {
							System.out.println("Unknown keyword: " + row[0]
									+ " on line " + lineCount);
							System.exit(1);
						}
					}
				} // end if(gotData)
			} // end while
			inr.close();
			//
			// Now we know everything to be known.
			// Construct the matrix from the file.
			// -----------------------------------
			int r = theData.size();
			int c = theData.get(0).length;
			if (ignoreVariables != null) {
			}
			Double[][] dataTmp = new Double[r][c];
			for (int i = 0; i < r; i++) {
				Double[] this_row = theData.get(i);
				for (int j = 0; j < c; j++) {
					dataTmp[i][j] = this_row[j];
				}
			}
			// Since we calculate the error on the variable we
			// must transpose the data matrix
			// -----------------------------------------------
			data = transposeMatrix(dataTmp);
		} catch (IOException e) {
			System.out.println(e);
			System.exit(1);
		}
	} // end readFile

	//
	// Transpose matrix
	// ----------------
	public static Double[][] transposeMatrix(Double[][] m) {
		int r = m.length;
		int c = m[0].length;
		Double[][] t = new Double[c][r];
		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				t[j][i] = m[i][j];
			}
		}
		return t;
	} // end transposeMatrix

	/*
	 * makeCommands: makes the CommandGene array given the function listed in
	 * the configurations file
	 * ------------------------------------------------------------
	 */
	static CommandGene[] makeCommands(GPConfiguration conf, String[] functions,
			Double lowerRange, Double upperRange, String type) {
		ArrayList<CommandGene> commandsList = new ArrayList<CommandGene>();
		int len = functions.length;
		try {
			for (int i = 0; i < len; i++) {
				//
				// Note: Not all functions are applicable here...
				//
				if ("Multiply".equals(functions[i])) {
					commandsList
							.add(new Multiply(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Multiply(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Multiply3".equals(functions[i])) {
					commandsList.add(new Multiply3(conf,
							CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Multiply3(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Add".equals(functions[i])) {
					commandsList.add(new Add(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList
								.add(new Add(conf, CommandGene.BooleanClass));
					}
				} else if ("Divide".equals(functions[i])) {
					commandsList.add(new Divide(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Divide(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Add3".equals(functions[i])) {
					commandsList.add(new Add3(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Add3(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Add4".equals(functions[i])) {
					commandsList.add(new Add4(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Add4(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Subtract".equals(functions[i])) {
					commandsList
							.add(new Subtract(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Subtract(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Sine".equals(functions[i])) {
					commandsList.add(new Sine(conf, CommandGene.DoubleClass));
				} else if ("ArcSine".equals(functions[i])) {
					commandsList
							.add(new ArcSine(conf, CommandGene.DoubleClass));
				} else if ("Tangent".equals(functions[i])) {
					commandsList
							.add(new Tangent(conf, CommandGene.DoubleClass));
				} else if ("ArcTangent".equals(functions[i])) {
					commandsList.add(new ArcTangent(conf,
							CommandGene.DoubleClass));
				} else if ("Cosine".equals(functions[i])) {
					commandsList.add(new Cosine(conf, CommandGene.DoubleClass));
				} else if ("ArcCosine".equals(functions[i])) {
					commandsList.add(new ArcCosine(conf,
							CommandGene.DoubleClass));
				} else if ("Exp".equals(functions[i])) {
					commandsList.add(new Exp(conf, CommandGene.DoubleClass));
				} else if ("Log".equals(functions[i])) {
					commandsList.add(new Log(conf, CommandGene.DoubleClass));
				} else if ("Abs".equals(functions[i])) {
					commandsList.add(new Abs(conf, CommandGene.DoubleClass));
				} else if ("Pow".equals(functions[i])) {
					commandsList.add(new Pow(conf, CommandGene.DoubleClass));
				} else if ("Round".equals(functions[i])) {
					commandsList.add(new Round(conf, CommandGene.DoubleClass));
				} else if ("Ceil".equals(functions[i])) {
					commandsList.add(new Ceil(conf, CommandGene.DoubleClass));
				} else if ("Floor".equals(functions[i])) {
					commandsList.add(new Floor(conf, CommandGene.DoubleClass));
				} else if ("Modulo".equals(functions[i])) {
					commandsList.add(new Modulo(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Modulo(conf,
								CommandGene.BooleanClass));
					}
				} else if ("ModuloD".equals(functions[i])) {
					commandsList
							.add(new ModuloD(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new ModuloD(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Max".equals(functions[i])) {
					commandsList.add(new Max(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList
								.add(new Max(conf, CommandGene.BooleanClass));
					}
				} else if ("Min".equals(functions[i])) {
					commandsList.add(new Min(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList
								.add(new Min(conf, CommandGene.BooleanClass));
					}
				} else if ("Sqrt".equals(functions[i])) {
					// Note: This uses my Sqrt.java file
					commandsList.add(new Sqrt(conf, CommandGene.DoubleClass));
				} else if ("Logistic".equals(functions[i])) {
					// Note: This uses my Logistic.java file
					commandsList
							.add(new Logistic(conf, CommandGene.DoubleClass));
				} else if ("Gaussian".equals(functions[i])) {
					// Note: This uses my Gaussian.java file
					commandsList
							.add(new Gaussian(conf, CommandGene.DoubleClass));
				} else if ("Sigmoid".equals(functions[i])) {
					// Note: This uses my Sigmoid.java file
					commandsList
							.add(new Sigmoid(conf, CommandGene.DoubleClass));
				} else if ("Gamma".equals(functions[i])) {
					// Note: This uses my Gamma.java file
					commandsList.add(new Gamma(conf, CommandGene.DoubleClass));
				} else if ("Step".equals(functions[i])) {
					// Note: This uses my Step.java file
					commandsList.add(new Step(conf, CommandGene.DoubleClass));
				} else if ("Sign".equals(functions[i])) {
					// Note: This uses my Sign.java file
					commandsList.add(new Sign(conf, CommandGene.DoubleClass));
				} else if ("Hill".equals(functions[i])) {
					// Note: This uses my Hill.java file
					commandsList.add(new Hill(conf, CommandGene.DoubleClass));
				} else if ("LesserThan".equals(functions[i])) {
					// experimental
					commandsList.add(new LesserThan(conf,
							CommandGene.BooleanClass));
				} else if ("GreaterThan".equals(functions[i])) {
					// experimental
					commandsList.add(new GreaterThan(conf,
							CommandGene.BooleanClass));
				} else if ("If".equals(functions[i])) {
					// Note: This is just If on DoubleClass, not a proper
					// Boolean
					commandsList.add(new If(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList
								.add(new If(conf, CommandGene.BooleanClass));
					}
				} else if ("IfElse".equals(functions[i])) {
					commandsList.add(new IfElse(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new IfElse(conf,
								CommandGene.BooleanClass));
					}
				} else if ("IfDyn".equals(functions[i])) {
					// Well, this don't work as expected...
					// System.out.println("IfDyn is not supported yet");
					commandsList.add(new IfDyn(conf, CommandGene.BooleanClass,
							1, 1, 5));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new IfDyn(conf,
								CommandGene.DoubleClass, 1, 1, 5));
					}
				} else if ("Loop".equals(functions[i])) { // experimental
					commandsList
							.add(new Loop(conf, CommandGene.DoubleClass, 3));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Loop(conf,
								CommandGene.BooleanClass, 3));
					}
				} else if ("Equals".equals(functions[i])) {
					// experimental
					// commandsList.add(new Equals(conf,
					// CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Equals(conf,
								CommandGene.BooleanClass));
					}
				} else if ("ForXLoop".equals(functions[i])) {
					// experimental
					commandsList.add(new ForXLoop(conf,
							CommandGene.IntegerClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new ForXLoop(conf,
								CommandGene.BooleanClass));
					} else if (useADF && "integer".equals(adfType)) {
						commandsList.add(new ForXLoop(conf,
								CommandGene.IntegerClass));
					}
				} else if ("ForLoop".equals(functions[i])) {
					// experimental
					commandsList.add(new ForLoop(conf,
							CommandGene.IntegerClass, 10));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new ForLoop(conf,
								CommandGene.BooleanClass, 10));
					} else if (useADF && "integer".equals(adfType)) {
						commandsList.add(new ForLoop(conf,
								CommandGene.IntegerClass, 10));
					}
				} else if ("Increment".equals(functions[i])) {
					commandsList.add(new Increment(conf,
							CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Increment(conf,
								CommandGene.BooleanClass));
					}
				} else if ("Argument".equals(functions[i])) {
					// experimental
					/*
					 * commandsList.add(new Argument(conf, 1,
					 * CommandGene.DoubleClass)); if ("boolean".equals(adfType))
					 * { commandsList.add(new Argument(conf, 1,
					 * CommandGene.BooleanClass)); }
					 */

				} else if ("StoreTerminal".equals(functions[i])) {
					// experimental
					commandsList.add(new StoreTerminal(conf, "dmem0",
							CommandGene.DoubleClass));
					commandsList.add(new StoreTerminal(conf, "dmem1",
							CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new StoreTerminal(conf, "bmem0",
								CommandGene.DoubleClass));
						commandsList.add(new StoreTerminal(conf, "bmem1",
								CommandGene.DoubleClass));
					}
				} else if ("Pop".equals(functions[i])) {
					// experimental
					// commandsList.add(new Pop(conf, CommandGene.DoubleClass));
					if (useADF && "boolean".equals(adfType)) {
						commandsList
								.add(new Pop(conf, CommandGene.BooleanClass));
					}
				} else if ("Push".equals(functions[i])) {
					// experimental
					commandsList.add(new Push(conf, CommandGene.DoubleClass));
				} else if ("And".equals(functions[i])) {
					// experimental
					commandsList.add(new And(conf));
				} else if ("Or".equals(functions[i])) {
					// experimental
					commandsList.add(new Or(conf));
				} else if ("Xor".equals(functions[i])) {
					// experimental
					commandsList.add(new Xor(conf));
				} else if ("Not".equals(functions[i])) {
					// experimental
					commandsList.add(new Not(conf));
				} else if ("AndD".equals(functions[i])) {
					// experimental
					commandsList.add(new AndD(conf));
				} else if ("OrD".equals(functions[i])) {
					// experimental
					commandsList.add(new OrD(conf));
				} else if ("XorD".equals(functions[i])) {
					// experimental
					commandsList.add(new XorD(conf));
				} else if ("NotD".equals(functions[i])) {
					// experimental
					commandsList.add(new NotD(conf));
				} else if ("SubProgram".equals(functions[i])) {
					// experimental
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new SubProgram(conf, new Class[] {
								CommandGene.BooleanClass,
								CommandGene.BooleanClass }));
						commandsList.add(new SubProgram(conf, new Class[] {
								CommandGene.BooleanClass,
								CommandGene.BooleanClass,
								CommandGene.BooleanClass }));
					}
					commandsList
							.add(new SubProgram(conf, new Class[] {
									CommandGene.DoubleClass,
									CommandGene.DoubleClass }));
					commandsList.add(new SubProgram(conf, new Class[] {
							CommandGene.DoubleClass, CommandGene.DoubleClass,
							CommandGene.DoubleClass }));
				} else if ("Tupel".equals(functions[i])) {
					// experimental
					if (useADF && "boolean".equals(adfType)) {
						commandsList.add(new Tupel(conf, new Class[] {
								CommandGene.BooleanClass,
								CommandGene.BooleanClass }));
					}
				} else {
					System.out.println("Unkown function: " + functions[i]);
					System.exit(1);
				}
			}
			commandsList.add(new Terminal(conf, CommandGene.DoubleClass,
					lowerRange, upperRange, terminalWholeNumbers));
			// commandsList.add(new Terminal(conf, CommandGene.BooleanClass,
			// lowerRange, upperRange, terminalWholeNumbers));

			// ADF
			// Just add the ADF to the "normal" command list (i.e. not to the
			// ADF list)
			if (useADF && !"ADF".equals(type)) {
				commandsList.add(new ADF(conf, 1, adfArity));
			}
			if (constants != null) {
				for (int i = 0; i < constants.size(); i++) {
					Double constant = constants.get(i);
					commandsList.add(new Constant(conf,
							CommandGene.DoubleClass, constant));
				}
			}
		} catch (Exception e) {
			System.out.println(e);
		}
		CommandGene[] commands = new CommandGene[commandsList.size()];
		commandsList.toArray(commands);
		return commands;
	}

	/**
	 * Starts the example.
	 * 
	 * @author Hakan Kjellerstrand
	 */
	public static void main(String[] args) throws Exception {
		// Use the log4j configuration
		// Log to stdout instead of file
		// -----------------------------
		// org.apache.log4j.PropertyConfigurator.configure("log4j.properties");
		LOGGER.addAppender(new ConsoleAppender(new SimpleLayout(), "System.out"));
		//
		// Read a configuration file, or not...
		//
		if (args.length > 0) {
			String filename = args[0];// e.g. "fahrenheit_celsius.conf"
			readFile(filename);
		}
		// Present the problem
		// -------------------
		System.out.println("Presentation: " + presentation);
		if (outputVariable == null) {
			outputVariable = numInputVariables;
		}
		if (variableNames == null) {
			variableNames = new String[numInputVariables + 1];
			for (int i = 0; i < numInputVariables + 1; i++) {
				variableNames[i] = "V" + i;
			}
		}
		System.out.println("output_variable: " + variableNames[outputVariable]
				+ " (index: " + outputVariable + ")");
		// Setup the algorithm's parameters.
		// ---------------------------------
		GPConfiguration config = new GPConfiguration();
		config.setGPFitnessEvaluator(new DeltaGPFitnessEvaluator());
		config.setMaxInitDepth(maxInitDepth);
		config.setPopulationSize(populationSize);
		// Default selectionMethod is is TournamentSelector(3)
		if (tournamentSelectorSize > 0) {
			config.setSelectionMethod(new TournamentSelector(
					tournamentSelectorSize));
		}
		/**
		 * The maximum depth of an individual resulting from crossover.
		 */
		config.setMaxCrossoverDepth(maxCrossoverDepth);
		config.setFitnessFunction(new XORGP.FormulaFitnessFunction());
		/**
		 * @param a_strict
		 *            true: throw an error during evolution in case a situation
		 *            is detected where no function or terminal of a required
		 *            type is declared in the GPConfiguration; false: don't
		 *            throw an error but try a completely different combination
		 *            of functions and terminals
		 */
		// config.setStrictProgramCreation(true);
		config.setStrictProgramCreation(false);
		// Default from GPConfiguration.java

		/**
		 * In crossover: If random number (0..1) < this value, then choose a
		 * function otherwise a terminal.
		 */
		config.setFunctionProb(functionProb);
		/**
		 * The probability that a reproduction operation is chosen during
		 * evolution. Must be between 0.0d and 1.0d. crossoverProb +
		 * reproductionProb must equal 1.0d.
		 */
		config.setReproductionProb(reproductionProb);
		/**
		 * The probability that a node is mutated during growing a program.
		 */
		config.setMutationProb(mutationProb);
		/**
		 * The probability that the arity of a node is changed during growing a
		 * program.
		 */
		config.setDynamizeArityProb(dynamizeArityProb);
		/**
		 * Percentage of the population that will be filled with new individuals
		 * during evolution. Must be between 0.0d and 1.0d.
		 */
		config.setNewChromsPercent(newChromsPercent);
		/**
		 * The minimum depth of an individual when the world is created.
		 */
		config.setMinInitDepth(minInitDepth);
		/**
		 * If m_strictProgramCreation is false: Maximum number of tries to
		 * construct a valid program.
		 */
		config.setProgramCreationMaxTries(programCreationMaxTries);
		XORGP problem = new XORGP(config);
		// Create the genotype of the problem, i.e., define the GP commands and
		// terminals that can be used, and constrain the structure of the GP
		// program.
		// --------------------------------------------------------------------
		GPGenotype gp = problem.create();
		// gp.setVerboseOutput(true);
		gp.setVerboseOutput(false);
		startTime = System.currentTimeMillis();
		// Start the computation with maximum 800 evolutions.
		// if a satisfying result is found (fitness value almost 0), JGAP stops
		// earlier automatically.
		// --------------------------------------------------------------------
		// gp.evolve(numEvolutions);

		//
		// I'm rolling my own to to be able to control output better etc.
		//
		System.out.println("Creating initial population");
		System.out.println("Mem free: "
				+ SystemKit.niceMemory(SystemKit.getTotalMemoryMB()) + " MB");
		IGPProgram fittest = null;
		double bestFit = -1.0d;
		int bestGen = 0;
		HashMap<String, Integer> similiar = null;
		if (showSimiliar) {
			similiar = new HashMap<String, Integer>();
		}
		for (int gen = 1; gen <= numEvolutions; gen++) {
			gp.evolve(); // evolve one generation
			gp.calcFitness();
			GPPopulation pop = gp.getGPPopulation();
			IGPProgram thisFittest = pop.determineFittestProgram();
			thisFittest.setApplicationData((Object) ("gen" + gen));
			ProgramChromosome chrom = thisFittest.getChromosome(0);
			String program = chrom.toStringNorm(0);
			double fitness = thisFittest.getFitnessValue();
			if (showSimiliar || showPopulation) {
				if (showPopulation) {
					System.out.println("Generation " + gen
							+ " (show whole population, sorted)");
				}
				pop.sortByFitness();
				for (IGPProgram p : pop.getGPPrograms()) {
					double fit = p.getFitnessValue();
					if (showSimiliar && fit <= bestFit) {
						String prog = p.toStringNorm(0);
						if (!similiar.containsKey(prog)) {
							similiar.put(prog, 1);
						} else {
							similiar.put(prog, similiar.get(prog) + 1);
						}
					}
					if (showPopulation) {
						String prg = p.toStringNorm(0);
						p.size();
						System.out.println("\tprogram: " + prg + " fitness: "
								+ fit);
					}
				}
			}
			//
			// Yes, I have to think more about this....
			// Right now a program is printed if it has
			// better fitness value than the former best solution.

			// if (gen % 25 == 0) {
			// myOutputSolution(fittest, gen);
			// }
			if (bestFit < 0.0d || fitness < bestFit) {
				bestGen = gen;
				myOutputSolution(thisFittest, gen);
				bestFit = fitness;
				fittest = thisFittest;
				if (showSimiliar) {
					// reset the hash
					similiar.clear(); // = new HashMap<String,Integer>();
				}
				// Ensure that the best solution is in the population.
				// gp.addFittestProgram(thisFittest);
			}
		}

		// Print the best solution so far to the console.
		// ----------------------------------------------
		// gp.outputSolution(gp.getAllTimeBest());

		System.out.println("\nAll time best (from generation " + bestGen + ")");
		myOutputSolution(fittest, numEvolutions);
		System.out.println("applicationData: " + fittest.getApplicationData());
		// Create a graphical tree of the best solution's program and write it
		// to
		// a PNG file.
		// ----------------------------------------------------------------------
		// problem.showTree(gp.getAllTimeBest(), "mathproblem_best.png");

		endTime = System.currentTimeMillis();
		long elapsedTime = endTime - startTime;
		System.out.println("\nTotal time " + elapsedTime + "ms");

		readFile(args[1]);
		FormulaFitnessFunction f = new FormulaFitnessFunction();
		double testSetResult = f.computeRawFitness(fittest);
		System.out.println("Fitness of best solution over test set:" + testSetResult);
		
		System.exit(0);
	}

	/**
	 * Fitness function for evaluating the produced fomulas, represented as GP
	 * programs. The fitness is computed by calculating the result (Y) of the
	 * function/formula for integer inputs 0 to 20 (X). The sum of the
	 * differences between expected Y and actual Y is the fitness, the lower the
	 * better (as it is a defect rate here).
	 */
	public static class FormulaFitnessFunction extends GPFitnessFunction {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;

		protected double evaluate(final IGPProgram a_subject) {
			return computeRawFitness(a_subject);
		}

		public double computeRawFitness(final IGPProgram ind) {
			// System.out.println("/////////COMPUTING FITNESS////////////");
			double error = 0.0f;
			Object[] noargs = new Object[0];
			// Evaluate function for the input numbers
			// --------------------------------------------
			// double[] results = new double[numRows];
			for (int j = 0; j < numRows; j++) {
				// Provide the variable X with the input number.
				// See method create(), declaration of "nodeSets" for where X is
				// defined.
				// -------------------------------------------------------------

				// set all the input variables
				int variableIndex = 0;
				for (int i = 0; i < numInputVariables + 1; i++) {
					if (i != outputVariable) {
						variables[variableIndex].set(data[i][j]);
						variableIndex++;
					}
				}
				try {
					double result = ind.execute_double(0, noargs);
					// System.out.println("RESULT = "+result + "EXPECTED" +
					// data[outputVariable][j]);
					// Sum up the error between actual and expected result to
					// get a defect
					// rate.
					// -------------------------------------------------------------------

					if (result <= 0 && data[outputVariable][j] == 4) {
						error++;
					} else if (result >= 0 && data[outputVariable][j] == 2) {
						error++;
					}

					// If the error is too high, stop evaluation and return
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
					System.out.println(ind);
					throw ex;
				}
			}
			/*
			 * // experimental ProgramChromosome chrom = ind.getChromosome(0);
			 * String program = chrom.toStringNorm(0); double length =
			 * program.length();
			 */
			error = (error / numRows) * 100;
			// If the fitness is very close to 0.0 then we maybe bump it
			// up to see alternative solutions.
			// -------------------------------------------------------
			if (error <= bumpValue && bumpPerfect) {
				if (!foundPerfect) {
					System.out.println("Found a perfect solution with err "
							+ error + "!. Bump up the values!");
					foundPerfect = true;
				}
				ProgramChromosome chrom = ind.getChromosome(0);
				String program = chrom.toStringNorm(0);
				if (!foundSolutions.containsKey(program)) {
					System.out.println("PROGRAM:" + program + " error: "
							+ error);
					foundSolutions.put(program, 1);
				} else {
					foundSolutions
							.put(program, foundSolutions.get(program) + 1);
				}
				error = 0.1d;
			}
			// a simplistc version of length punishing
			/*
			 * // too experimental if (punishLength) { return error + length; }
			 * else { return error; }
			 */

			if (scaleError > 0.0d) {
				return error * scaleError;
			} else {
				return error;
			}
		}
	}

	/**
	 * Outputs the best solution until now at standard output.
	 * 
	 * This is stolen (and somewhat edited) from GPGenotype.outputSolution which
	 * used log4j.
	 * 
	 * @param a_best
	 *            the fittest ProgramChromosome
	 * 
	 * @author Hakan Kjellerstrand (originally by Klaus Meffert)
	 */
	public static void myOutputSolution(IGPProgram a_best, int gen) {
		String freeMB = SystemKit.niceMemory(SystemKit.getFreeMemoryMB());
		System.out.println("Evolving generation " + (gen) + "/" + numEvolutions
				+ ", memory free: " + freeMB + " MB");
		if (a_best == null) {
			System.out.println("No best solution (null)");
			return;
		}
		double bestValue = a_best.getFitnessValue();
		if (Double.isInfinite(bestValue)) {
			System.out.println("No best solution (infinite)");
			return;
		}
		System.out.println("Best solution fitness: "
				+ NumberKit.niceDecimalNumber(bestValue, 2) + " % error");
		System.out.println("Best solution: " + a_best.toStringNorm(0));
		String depths = "";
		int size = a_best.size();
		for (int i = 0; i < size; i++) {
			if (i > 0) {
				depths += " / ";
			}
			depths += a_best.getChromosome(i).getDepth(0);
		}
		if (size == 1) {
			System.out.println("Depth of chrom: " + depths);
		} else {
			System.out.println("Depths of chroms: " + depths);
		}
	}
}
 abstract class GPProblem {
	  /** String containing the CVS revision. Read out via reflection!*/
	  private final static String CVS_REVISION = "$Revision: 1.8 $";

	  private GPConfiguration m_conf;

	  public GPProblem(GPConfiguration a_conf)
	      throws InvalidConfigurationException {
	    if (a_conf == null) {
	      throw new InvalidConfigurationException("Configuration must not be null!");
	    }
	    m_conf = a_conf;
	  }

	  /**
	   * Default constructor for dynamic instantiation.
	   *
	   * @author Klaus Meffert
	   * @since 3.2
	   */
	  public GPProblem() {

	  }

	  /**
	   * @return newly created GPGenotype
	   * @throws InvalidConfigurationException
	   *
	   * @author Klaus Meffert
	   * @since 3.0
	   */
	  public abstract GPGenotype create()
	      throws InvalidConfigurationException;

	  /**
	   * Creates a graphical tree out of a given GP program and saves it to a file.
	   *
	   * @param a_prog the GP program to visualize a tree for
	   * @param a_filename the name of the file to save the tree in
	   * @throws InvalidConfigurationException
	   *
	   * @author Klaus Meffert
	   * @since 3.0
	   */
	  public void showTree(IGPProgram a_prog, String a_filename)
	      throws InvalidConfigurationException {
	    if (a_prog == null) {
	      return;
	    }
	    TreeNode myTree = createTree(a_prog);
	    if (myTree == null) {
	      return;
	    }
	    TreeVisualizer tv = new TreeVisualizer();
	    tv.setTreeBranchRenderer(new JGAPTreeBranchRenderer());
	    tv.setTreeNodeRenderer(new JGAPTreeNodeRenderer());
	    tv.setBranchStartWidth(18.0);
	    tv.setArenaColor(Color.black);
	    tv.setBkgndColor(Color.black);
	    tv.setRenderNodes(true);
	    tv.setSide(1024);
	    tv.setCircleDiminishFactor(0.5);
	    tv.writeImageFile(tv.renderTree(myTree), new File(a_filename));
	  }

	  /**
	   * Creates a tree out of a given GP program and saves it to a file. Allows to
	   * preset the tree renderers.
	   *
	   * @param a_prog the GP program to visualize a tree for
	   * @param a_filename the name of the file to save the tree in
	   * @param a_treeBranchRenderer renderer for the tree's branches
	   * @param a_treeNodeRenderer renderer for the tree's nodes
	   * @throws InvalidConfigurationException
	   *
	   * @author Klaus Meffert
	   * @since 3.0
	   */
	  public void showTree(IGPProgram a_prog, String a_filename,
	                       TreeBranchRenderer a_treeBranchRenderer,
	                       TreeNodeRenderer a_treeNodeRenderer)
	      throws InvalidConfigurationException {
	    TreeNode myTree = createTree(a_prog);
	    if (myTree == null) {
	      return;
	    }
	    TreeVisualizer tv = new TreeVisualizer();
	    tv.setTreeBranchRenderer(a_treeBranchRenderer);
	    tv.setTreeNodeRenderer(a_treeNodeRenderer);
	    tv.setBranchStartWidth(18.0);
	    tv.setArenaColor(Color.black);
	    tv.setBkgndColor(Color.black);
	    tv.setRenderNodes(true);
	    tv.setSide(1024);
	    tv.setCircleDiminishFactor(0.5);
	    tv.writeImageFile(tv.renderTree(myTree), new File(a_filename));
	  }

	  /**
	   * Creates a tree out of a given GP program.
	   *
	   * @param a_prog the GPGenotype to visualize a tree for
	   * @return the TreeNode object corresponding to the GP program
	   * @throws InvalidConfigurationException
	   *
	   * @author Klaus Meffert
	   * @since 3.0
	   */
	  public TreeNode createTree(IGPProgram a_prog)
	      throws InvalidConfigurationException {
	    if (a_prog == null) {
	      return null;
	    }
	    ProgramChromosome master = new ProgramChromosome(m_conf);
	    master.setIndividual(a_prog);
	    TreeNode tree;
	    if (a_prog.size() > 1) {
	      Class[] types = new Class[a_prog.size()];
	      for (int i = 0; i < a_prog.size(); i++) {
	        types[i] = CommandGene.VoidClass; //this is arbitrary
	      }
	      master.setGene(0, new SubProgram(m_conf, types));
	      int index = 1;
	      for (int i = 0; i < a_prog.size(); i++) {
	        ProgramChromosome child = a_prog.getChromosome(i);
	        for (int j = 0; j < child.size(); j++) {
	          master.setGene(index++, child.getGene(j));
	        }
	      }
	      master.redepth();
	      tree = new JGAPTreeNode(master, 0);
	    }
	    else {
	      tree = new JGAPTreeNode(a_prog.getChromosome(0), 0);
	    }
	    return tree;
	  }

	  /**
	   * @return the GPConfiguration set
	   *
	   * @author Klaus Meffert
	   * @since 3.0
	   */
	  public GPConfiguration getGPConfiguration() {
	    return m_conf;
	  }

	  /**
	   * Sets the configuration. Only use in case of dynamic instantiation (in case
	   * constructor with parameter GPConfiguration is not used).
	   *
	   * @param a_conf the configuration to set
	   *
	   * @author Klaus Meffert
	   * @since 3.2
	   */
	  protected void setGPConfiguration(GPConfiguration a_conf) {
	    m_conf = a_conf;
	  }
	}
