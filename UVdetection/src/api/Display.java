package api;

import java.awt.Image;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

public class Display {
	static final int screenWidth = 1920;
	static final int screenHeight = 1080;
	static final int displayDivide = 2;
	static int x,y = 0;
	
	public static void show(image image) {
		Mat data = image.me;
		
		//positioning
		if(x+data.width()/displayDivide>screenWidth-40) {
			x=0;
			y+=data.height()/displayDivide+20;
		}
		
		//resizeWindow() is destructive! wtf?
		String hashname = image.name+String.valueOf(Math.random());
		
		HighGui.imshow(hashname, data.clone());
		HighGui.resizeWindow(hashname, 
				data.width()/displayDivide, 
				data.height()/displayDivide);
		HighGui.moveWindow(hashname, x, y);
		
		x+=data.width()/displayDivide+20;
	}
}
