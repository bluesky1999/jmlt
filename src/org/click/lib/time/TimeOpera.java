package org.click.lib.time;

import java.sql.Timestamp;
import java.text.ParsePosition;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.click.lib.string.SSO;

/**
 * 时间处理的工具类
 * 
 * @author zkyz
 *
 */
public class TimeOpera {

	public static final long PERIOD_DAY = 24 * 60 * 60 * 1000;

	public static final long PERIOD_MINUTE = 60 * 1000;

	public static final long PERIOD_HOUR = 60 * 60 * 1000;

	public static int formatDay(Date date) {

		return 0;
	}

	/**
	 * from format:yyyy-MM-dd HH:mm:ss to long
	 * 
	 * @param str
	 * @return
	 */
	public static long str2long(String str) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date dt = null;
		try {
			dt = sdf.parse(str);
		} catch (Exception e) {

		}
		if (dt == null) {
			return -1;
		}

		return dt.getTime();
	}

	/**
	 * from format:yyyy-MM-dd HH:mm:ss to long
	 * 
	 * @param str
	 * @return
	 */
	public static long str2long2(String str) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd HH:mm:ss");
		Date dt = null;
		try {
			dt = sdf.parse(str);
		} catch (Exception e) {

		}
		if (dt == null) {
			return -1;
		}

		return dt.getTime();
	}

	public static String long2strm(long tl) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Timestamp ts = new Timestamp(tl * 1000);
		//Date dt = new Date(tl);	
		return sdf.format(ts);
	}

	public static String int2string(String intday) {
		if (intday.length() != 8) {
			return "";
		}

		String year = intday.substring(0, 4);
		String month = intday.substring(4, 6);
		String day = intday.substring(6, 8);

		return year + "-" + month + "-" + day;
	}

	public static String long2str(long tl) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date dt = new Date(tl);
		return sdf.format(dt);
	}

	public static String getDateFromStr(String str) {
		String dstr = "";

		String date_regex = "([\\d]+\\-[\\d]+\\-[\\d]+)";
		Pattern date_pat = Pattern.compile(date_regex);
		Matcher date_mat = date_pat.matcher(str);
		if (date_mat.find()) {
			dstr = date_mat.group(1);
		}
		if (!(SSO.tnoe(dstr))) {
			dstr = "";
		}
		dstr = dstr.replaceAll("\\-", "");
		dstr = dstr.trim();

		return dstr;
	}

	public static String getTimeFromStr(String str) {
		String tstr = "";

		String time_regex = "([\\d]+:[\\d]+:[\\d]+)";
		Pattern time_pat = Pattern.compile(time_regex);
		Matcher time_mat = time_pat.matcher(str);
		if (time_mat.find()) {
			tstr = time_mat.group(1);
		}
		if (!(SSO.tnoe(tstr))) {
			tstr = "";
		}
		tstr = tstr.trim();

		return tstr;
	}

	/**
	 * 获得当前日期，格式20140626
	 * 
	 * @return
	 */
	public static int getToday() {
		long ctime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");
		Date dt = new Date(ctime);

		return Integer.parseInt(sdf.format(dt));
	}

	/**
	 * 获得当前日期，格式20140626
	 * 
	 * @return
	 */
	public static int getHour() {
		long ctime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("HH");
		Date dt = new Date(ctime);

		return Integer.parseInt(sdf.format(dt));
	}

	/**
	 * 获得当前日期，格式2014-06-26
	 * 
	 * @return
	 */
	public static String getTodayStr() {
		long ctime = System.currentTimeMillis();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
		Date dt = new Date(ctime);

		return sdf.format(dt);
	}

	/**
	 * 获得当前日期，格式20140626
	 * 
	 * @return
	 */
	public static int getYesterday() {
		long ctime = System.currentTimeMillis();
		ctime = ctime - PERIOD_DAY;
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");
		Date dt = new Date(ctime);

		return Integer.parseInt(sdf.format(dt));
	}

	public static String getOnedayBefore(String time_str) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date dt = null;
		try {
			dt = sdf.parse(time_str);
		} catch (Exception e) {

		}
		long ctime = dt.getTime() - PERIOD_DAY;

		return long2str(ctime);
	}

	public static String getOnedayAfter(String time_str) {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date dt = null;
		try {
			dt = sdf.parse(time_str);
		} catch (Exception e) {

		}
		long ctime = dt.getTime() + PERIOD_DAY;

		return long2str(ctime);
	}

	public static String getCurrentTime() {
		long ctime = System.currentTimeMillis();
		String cstr = long2str(ctime);
		//System.out.println("cstr:"+cstr);
		return cstr;
	}

	public static long getCurrentTimeLong() {
		long ctime = System.currentTimeMillis();
		return ctime;
	}

	public static String getDaySplit(String time) {
		long tl = str2long2(time);
		SimpleDateFormat sdf = new SimpleDateFormat("HH");
		Date dt = new Date(tl);

		int hour = Integer.parseInt(sdf.format(dt));

		String ds = "";

		if (hour > 23 || hour < 7) {
			ds = "deep";
		} else if (hour > 19 && hour <= 23) {
			ds = "night";
		} else {
			ds = "day";
		}

		return ds;
	}

	/**
	 * 根据一个日期，返回是星期几的字符串
	 * 
	 * @param sdate
	 * @return
	 */
	public static String getWeek(String sdate) {
		// 再转换为时间
		Date date = strToDate(sdate);
		Calendar c = Calendar.getInstance();
		c.setTime(date);
		// int hour=c.get(Calendar.DAY_OF_WEEK);
		// hour中存的就是星期几了，其范围 1~7
		// 1=星期日 7=星期六，其他类推
		return new SimpleDateFormat("EEEE").format(c.getTime());
	}

	/**
	 * 将短时间格式字符串转换为时间 yyyy-MM-dd
	 * 
	 * @param strDate
	 * @return
	 */
	public static Date strToDate(String strDate) {
		SimpleDateFormat formatter = new SimpleDateFormat("yyyyMMdd");
		ParsePosition pos = new ParsePosition(0);
		Date strtodate = formatter.parse(strDate, pos);
		return strtodate;
	}

	public static void main(String[] args) {
		/*
		String[] dstr={"2014-02-08 15:29:05","2014-02-08 19:29:20","2014-02-08 02:22:36","2014-02-08 15:04:25","2014-02-08 14:22:40","2014-02-08 16:17:22","2014-02-08 09:28:47"};
		for(int i=0;i<dstr.length;i++)
		{
			System.out.println(i+"   "+getTimeFromStr(dstr[i])+"    "+getDateFromStr(dstr[i]));
			
			System.out.println(i+":"+str2long(dstr[i])+"    "+long2str(str2long(dstr[i])-999));
		}
		*/
		//for(int i=0;i<10;i++)
		//System.out.println(getCurrentTimeLong());
		//System.out.println(getOnedayBefore("2014-02-08 19:29:20"));
		//System.out.println(getHour());

		/*
		System.out.println(long2strm(1428187191));
		System.out.println(long2strm(1428187154));
		System.out.println(long2strm(1428187180));
		System.out.println(long2strm(1428187090));
		
		//System.out.println(long2str(1391858960000));
		System.out.println(str2long("2014-02-08 19:29:20"));
		System.out.println(str2long("2014-02-08 15:29:05"));
		System.out.println(str2long("2014-02-08 14:22:40"));
		
		
		System.out.println("c:"+System.currentTimeMillis());
		String astr="2015-04-05";
		System.out.println("astr.len:"+astr.length());
		
		System.out.println("conv:"+int2string("20150405"));
		*/

		System.out.println(getDaySplit("20151021 01:44:59"));
		System.out.println(getDaySplit("2014-02-08 03:29:20"));
		System.out.println(getDaySplit("2014-02-08 15:29:05"));

		System.out.println("20151108:"+getWeek("20151108"));
		System.out.println("20151109:"+getWeek("20151109"));
		System.out.println("20151110:"+getWeek("20151110"));
		System.out.println("20151111:"+getWeek("20151111"));
		System.out.println("20151112:"+getWeek("20151112"));
		System.out.println("20151113:"+getWeek("20151113"));	
		System.out.println("20151114:"+getWeek("20151114"));
		System.out.println("20151115:"+getWeek("20151115"));
		System.out.println("20151116:"+getWeek("20151116"));
	}

	public static long getEntireDay() {
		long entireDay = 1000 * 60 * 60 * 24;
		return entireDay;
	}

}
