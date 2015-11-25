package org.click.lib.codedes;

public class Swap {
    private static String key = "test";

    private static String swap(String str) {
            String res = "";
            StringBuilder sb = new StringBuilder(str);

            int temp = 0;

            for (int i = 0; i < str.length(); i++) {
                    if (i % 7 == 0) {
                            temp = sb.charAt(i);
                            sb.setCharAt(i, (char) (temp + 5));
                    }
            }

            res = sb.toString();

            return res;
    }

    private static String reswap(String str) {
            String res = "";
            StringBuilder sb = new StringBuilder(str);

            int temp = 0;

            for (int i = 0; i < str.length(); i++) {
                    if (i % 7 == 0) {
                            temp = sb.charAt(i);
                            sb.setCharAt(i, (char) (temp - 5));
                    }
            }

            res = sb.toString();

            return res;
    }

    private static String encode(String str) {
            String encode = "";

            try {
                    encode = DesUtil.encrypt(str, swap(key));
            } catch (Exception e) {

            }
            encode = encode.replaceAll("\\s+", "");

            return encode;
    }

    private static String decode(String str) {
            String decode = "";

            try {
                    decode = DesUtil.decrypt(str, swap(key));
            } catch (Exception e) {

            }
            return decode;
    }
    
    public static void main(String[] args)
    {
    	Swap sw=new Swap();
    	String encode="zVdKg6yO+6hHTgKN1riaa2MwLgvIGFTFISz3WMqjQ/KoQFFy4X31ZpzyLbeKT6LIsrGbvstVIO8nr+qR6bM+KYrKT3Dgdkr/WFJbctVUiw/3LkxZ5pnUV/XFUECKz/e85izIN0FZIFetf+zlGJKA4IB3+quYxwlZRnwXvRLsXD9pb4CZd96KDCVsvOWwTGy1Hiev7OjylLD5rIf7kGVF4NwlI9vD43VA9tlxa7bIRxnWfo9vuYbdyWIY+WEajKQ6mWjpnKSP41upTPKvx8oVzjT7g7WStae5wpo1pRfZn7eeq6rUGK4puBZLeSCtEVcHq1oGtLIvpHgOpwr4Jvj3sdgmZiS1a9USI2GUw+IRj5wnUIyD1BI6eqKD9XXbuBw951cRzSS9R+QaT70dha/Plge8arPq9LMxWeyXP3kAJ6eWGZqcIBBHnNeeSPviWE1u2oOnMpeofn7wjbKlvb9w0nUta9bzIyok7bWVWWsxS4aGUo4ZUe7R3b2SefULpjGq85A5E0IpADPgG87c0fBpUNUHfI5as1Qt1JBEKeaosThrod+d5Rh38rjMEgR3Rfbvc8DfvVWgT82lUDqMistN7EYOYlxoW/v86f4Byx86YvT9zYDdhXqnUafAEJEmsACegVhoRr3vNBPtpaJjPk9aU794KygmESm5uDrjod6gSy2y4xwZGz7xubpO2H6bvqZPB/G37oRYdrvH5LTzMiPgZ1kgA6EERz3GF2riZ4UhS1YDTDbJUWnoJIHwaZRO1zAWi76IyM11XhWL1d2nFHbaBHsbvEXEKw3tDfDB0ePCqVFmYKJGenYqPZcsodfEyLK5He+AsGbkziMl9+8sI4ApxNzkXpr+nBWAJNE1fU0J2CU8f1yYn9PpExWifTtQUnfdDrBZGbObZNKisa9D+RIwy60zku0emzzkiLsJYnAwPl2agrqTpqexJGXFOwQ0VEUsgSwWflJ7IkRp9LRGxaI8gJiN9JgqO0DGm3GtWVqAiiVe8TSLVjQD+TE4p53uFv2YtUfIq390+y8PgHKvjG1YpvIwsN9FGGrpDNXBw/2zNI2nVoGpevOcRTLPiOEcPKxoif5naFVoYe79OIUf9i/ICcSzS2WlN63X6FPyWth0m6ZIkEWL8WE7dBgg5VF76kvoiGSsrJOKvES/sg6WkshlePE96pyFZnol5kP0RC+gYU2HsK8qRko+ysj5k8ogYBpudVKYTXZ3RgV02Fq9m3TtVeeEc1OnBU9dyBAzHzKRho71D0UCCvz3Cci6vKdd89J+7Pway+ySlrjI3AJuxAXABncCv2rKv7AGEoHZIbGBt1zqXtaA2X5/pajPN6ZjMLsEXXtzdE2hnvLHdCJgc74xtmoJQNHUjsgPHj8F6D0DEwywRaG+XiB8AUnp3DSoq/DgIeuAaouQcRQ=";
    	
    	String decode="";
    	
    	decode=sw.decode(encode);
    	
    	System.err.println("decode:"+decode);
    	
    }
}
