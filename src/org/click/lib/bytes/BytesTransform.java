package org.click.lib.bytes;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;

//byte array into various values
public class BytesTransform {

	/**
	 * byte array convert to int
	 * @param b
	 * @return
	 */
	public static int byteToInt2(byte[] b) {

		int i = 0;
		try {
			ByteArrayInputStream bintput = new ByteArrayInputStream(b);
			DataInputStream dintput = new DataInputStream(bintput);
			i = dintput.readInt();
			bintput.close();
			dintput.close();
			bintput = null;
			dintput = null;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return i;
	}

	/**
	 * byte array convert to int another version
	 * @param b
	 * @return
	 */
	public static int byteToIntv(byte[] b) {

		int mask = 0xff;
		int temp = 0;
		int n = 0;
		for (int i = 0; i < b.length; i++) {
			n <<= 8;
			temp = b[i] & mask;
			n |= temp;
		}

		return n;

	}

	/**
	 * byte array convert to string
	 * @param b
	 * @return
	 */
	public static String bytes2str(byte[] b) {
		String str = "";
		for (int i = 0; i < b.length; i++) {
			int v = b[i] & 0xFF;
			String hv = Integer.toHexString(v);
			if (hv.length() == 1) {
				hv = '0' + hv;
			}
			str += (hv + " ");
		}
		return str;
	}

	/**
	 * complete bytes to four bytes, bytes not reach four bytes is set to 0 from low end
	 * @param b
	 * @return
	 */
	public static byte[] completeBytes(byte[] b) {
		byte[] cb = new byte[4];
		for (int i = 0; i < 4 - b.length; i++) {
			cb[i] = 0;
		}

		for (int i = 4 - b.length; i < 4; i++) {
			cb[i] = b[i + b.length - 4];
		}

		return cb;
	}



}
