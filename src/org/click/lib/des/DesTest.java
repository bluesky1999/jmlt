package org.click.lib.des;

public class DesTest {

	private static String key="test";

	public static String encode(String data) {

		String encode = "";
		try {
			encode =  DesUtil.encrypt(data, key);
		} catch (Exception e) {

		}
		encode = encode.replaceAll("\\s+", "");
		return encode;
	}

	
	public static String decode(String data) {
	
		String decode = "";
		try {
			decode = DesUtil.decrypt(data, key);
		} catch (Exception e) {

		}
		return decode;
	}
	
}
