package Main;

import org.opencv.core.Core;
import org.opencv.dnn.Dnn;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;

import api.image;

public class UVDetectionCore {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		//about 31 milliseconds per cycle
		double start = System.currentTimeMillis();
		for(int i=0; i<20; i++) {
			process("UV.jpg","noUV.jpg",20,25);
			//process("UV1.jpg","noUV1.jpg",-7,-10);
		}
		double total = System.currentTimeMillis()-start;
		System.out.println(total/20);
		
		//OpenCV does bad job deleting windows
		//So we must ourselves through catastrophic means
		//HighGui.waitKey(0);
		System.exit(69);
	}
	
	
	static void process(String wUV, String woUV, int xTrans, int yTrans) {
		
		var UV = new image(wUV,"With UV Light");
		var noUV = new image(woUV,"Without UV Light");
		
			//UV.show();
			//noUV.show();
		
		//noise Map
		UV.findEdges();
		noUV.findEdges();
		
		UV.translate(xTrans, yTrans);
			//noUV.show();
			//UV.show();
		
		UV.subtract(noUV);
		
		//remove low frq noise
		var UVmask = UV.clone();
		UVmask.binarize(16); //bottleneck
		UVmask.invert();		
		UV.subtract(UVmask);

			//UV.show();
		
			//find shapes
		var contours = UV.contours(false);
		image.sortContours(contours);
		
		var display = new image(wUV,"With UV Light");
		display.translate(xTrans,yTrans);
		display.drawContours(contours);
		//display.show();
		
	}
}
