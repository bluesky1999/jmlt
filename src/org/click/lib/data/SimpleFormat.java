package org.click.lib.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.click.lib.sort.SortStrArray;
import org.click.lib.string.SSO;

public class SimpleFormat extends Format {

	@Override
	public List<WORD> processLine(String words) {
		List<WORD> wordList = new ArrayList<WORD>();
		try {
			String[] tokens = words.split(";");

			String token = "", key = "", value = "";

			if (tokens == null) {
				return null;
			}

			WORD word = null;

			HashMap<String, Double> wordHash = new HashMap<String, Double>();
			HashMap<String, Integer> wordCount = new HashMap<String, Integer>();

			double val = 0.0;

			for (int j = 0; j < tokens.length; j++) {

				token = tokens[j];
				if (SSO.tioe(token) || token.trim().equals("-")) {
					continue;
				}

				if (token.split(":").length != 2) {
					continue;
				}

				key = token.split(":")[0].trim();
				value = token.split(":")[1].trim();

				// word = new WORD(key + "", Double.parseDouble(value));
				// wordList.add(word);
				val = Double.parseDouble(value);

				if (!(wordHash.containsKey(key))) {
					wordHash.put(key, val);
					wordCount.put(key, 1);
				} else {
					wordHash.put(key, wordHash.get(key) + val);
					wordCount.put(key, wordCount.get(key) + 1);
				}

			}

			int count = 0;
			for (Map.Entry<String, Double> entry : wordHash.entrySet()) {
				count = wordCount.get(entry.getKey());
				if (count < 1) {
					continue;
				}

				word = new WORD(entry.getKey() + "", entry.getValue()
						/ (double) count);
				wordList.add(word);
			}

		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

		return wordList;
	}

	@Override
	public String genLineSample(String line) {

		int labelIndex;
		int wordIndex;
		ArrayList<String> wordSimpleList = null;

		String[] sortArr = null;

		String formatLine = "";

		List<WORD> wordList = null;
		WORD word = null;

		if (SSO.tioe(line)) {
			return "";
		}

		wordList = processLine(line);

		if (wordList == null || wordList.size() < 1) {
			return "";
		}

		labelIndex = -1;

		// formatLine = labelIndex + " ";
		formatLine = "";
		wordSimpleList = new ArrayList<String>();
		for (int k = 0; k < wordList.size(); k++) {
			word = wordList.get(k);
			if (wordDict.containsKey(word.key)) {
				wordIndex = wordDict.get(word.key);
				wordSimpleList.add(wordIndex + "\001" + word.value);
			}
		}

		sortArr = SortStrArray.sort_List(wordSimpleList, 0, "int", 2, "\001");

		for (int k = sortArr.length - 1; k >= 0; k--) {
			formatLine += (sortArr[k].split("\001")[0] + ":"
					+ sortArr[k].split("\001")[1] + " ");
		}

		formatLine = formatLine.trim();

		return formatLine;

	}

	public void random(String input, String output) {
		try {

			ArrayList<String> list = new ArrayList<String>();

			BufferedReader br = new BufferedReader(new FileReader(input));
			PrintWriter pw = new PrintWriter(output);

			String line = "";

			while ((line = br.readLine()) != null) {
				if (SSO.tioe(line)) {
					continue;
				}

				line = line.trim();
				list.add(line);
			}

			br.close();

			for (int i = 0; i < list.size(); i++) {
				swap(list);
			}

			for (int i = 0; i < list.size(); i++) {
				pw.println(list.get(i));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void swap(ArrayList<String> list) {

		int firstId = 0;
		int secondId = 0;

		firstId = (int) (Math.random() * (list.size() - 1));
		secondId = (int) (Math.random() * (list.size() - 1));

		String first = list.get(firstId);
		String second = list.get(secondId);

		list.set(firstId, second);
		list.set(secondId, first);
	}

	@Override
	public void loadDict(String word_dict, String label_dict) {

		try {

			// BufferedReader wbr = new BufferedReader(new
			// FileReader(word_dict));
			BufferedReader wbr = null;
			InputStream inputStream = null;
			try {
				inputStream = this.getClass().getResourceAsStream(word_dict);
				System.err.println("word_dict:" + word_dict + " label_dict:"
						+ label_dict);

				wbr = new BufferedReader(new InputStreamReader(inputStream));
			} catch (Exception e) {
				wbr = new BufferedReader(new FileReader(word_dict.substring(1,
						word_dict.length())));
			}

			String line = "", key = "";
			int value = 0;
			String[] tokens = null;

			while ((line = wbr.readLine()) != null) {
				if (SSO.tioe(line)) {
					continue;
				}

				line = line.trim();
				tokens = line.split("\001");

				if (tokens == null || tokens.length != 2) {
					continue;
				}

				key = tokens[0];
				value = Integer.parseInt(tokens[1]);

				if (SSO.tioe(key)) {
					continue;
				}

				wordDict.put(key, value);
			}

			try {
				inputStream.close();
				wbr.close();
			} catch (Exception e) {

			}
			// BufferedReader lbr = new BufferedReader(new
			// FileReader(label_dict));
			BufferedReader lbr = null;
			InputStream linputStream = null;
			try {
				linputStream = this.getClass().getResourceAsStream(label_dict);

				lbr = new BufferedReader(new InputStreamReader(linputStream));
			} catch (Exception e) {
				lbr = new BufferedReader(new FileReader(label_dict.substring(1,
						label_dict.length())));
			}
			while ((line = lbr.readLine()) != null) {

				if (SSO.tioe(line)) {
					continue;
				}

				line = line.trim();
				tokens = line.split("\001");

				if (tokens == null || tokens.length != 2) {
					continue;
				}

				key = tokens[0];
				value = Integer.parseInt(tokens[1]);

				if (SSO.tioe(key)) {
					continue;
				}

				labelDict.put(key, value);
				labelIndex.put(value, key);
			}

			try {
				linputStream.close();
				lbr.close();
			} catch (Exception e) {

			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {

		SimpleFormat tf = new SimpleFormat();

		String dir = "dratio_admaster_letv_join/pre_gender";

		tf.buildDict(dir + "/preprocess_gender.txt", dir + "/words.txt", dir
				+ "/label.txt");

		// tf.loadDict("/dratio_letv_join_cvm3/dl_train_gender/words.txt","/dratio_letv_join_cvm3/dl_train_gender/label.txt");
		// tf.loadDict(dir+"/words.txt", dir+"/label.txt");
		tf.genSample(dir + "/preprocess_gender.txt", dir + "/format.txt");
		// tf.random("admaster_letv/format.txt",
		// "admaster_letv/format_rand.txt");
		tf.genSampleTrainTest(dir + "/format.txt", dir + "/train.txt", dir
				+ "/test.txt");

		// tf.featureCount("admaster_letv/preprocess.txt",
		// "admaster_letv/featureAnalysis.txt");
	}

}
