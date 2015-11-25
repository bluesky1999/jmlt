package org.click.bug;

public class CmdTest {

	public static void main(String[] args)
	{
		 String bzip2 = System.getProperty("bzip2", "bzip2");
		 System.out.println("bzip2:"+ bzip2);
	}
}
