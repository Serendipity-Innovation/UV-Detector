package Main;

import org.opencv.core.Core;
import org.opencv.dnn.Dnn;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;

import api.image;

public class UVDetectionCore {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//about 26 milliseconds per cycle
		double start = System.currentTimeMillis();
		for(int i=0; i<200; i++)process();
		double total = System.currentTimeMillis()-start;
		System.out.println(total/200);
		
		//OpenCV does bad job deleting windows
		//So we must ourselves through catastrophic means
		//HighGui.waitKey();
		System.exit(69);
	}
	
	
	static void process() {
		
		var UV = new image("UV.jpg","With UV Light");
		var noUV = new image("noUV.jpg","Without UV Light");
		
			//UV.show();
			//noUV.show();
		
		UV.findEdges();
		noUV.findEdges();
		
		UV.binarize(60);
		noUV.binarize(120);
		
		UV.removeSpecks(3);
		noUV.removeSpecks(2);
				
		UV.translate(20,25);
			
		//UV image is ALWAYS smoother than noUV so noUV ALWAYS has more edges
		//Therefore it is okay to subtract away edges from UV
		noUV.dilate(10);
		UV.subtract(noUV);
		
		var contours = UV.contours(false);
		image.sortContours(contours);
		UV.toColor();
		UV.drawContours(contours);
		
		//UV.show();
		
	}
}
