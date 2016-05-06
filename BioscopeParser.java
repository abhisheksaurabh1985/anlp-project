/*
	This program parses the bioscope data in an xml file, it removes all the tags and prints out the bare sentence into a text file. 
	The program is run in the following way:
		javac BioscopeParser.java
		java Bioscope <name of the xml file>
	It will parse the xml file for example 'file.xml' and print out the data in a text file
	named 'file.txt'
*/

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.*;


public class BioscopeParser {


   public static void main(String argv[]) throws IOException{

      try {

	 String fileName = argv[0];

         File inputFile = new File(fileName);
	 String outputFile = fileName.substring(0,fileName.indexOf('.'));
	 outputFile = outputFile+".txt";
	 PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
	/*		 String savestr = "mysave.sav";
		File f = new File(savestr);

		PrintWriter out = null;
		if ( f.exists() && !f.isDirectory() ) {
		    out = new PrintWriter(new FileOutputStream(new File(savestr), true));
		    out.append(mapstring);
		    out.close();
		}
		else {
		    out = new PrintWriter(savestr);
		    out.println(mapstring);
		    out.close();
		}
	 */
         DocumentBuilderFactory dbFactory =
            DocumentBuilderFactory.newInstance();
         DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
         Document doc = dBuilder.parse(inputFile);
         doc.getDocumentElement().normalize();
         System.out.print("Root element: ");
         System.out.println(doc.getDocumentElement().getNodeName());
	 NodeList s = doc.getElementsByTagName("sentence");
	 for(int i=0; i<s.getLength(); i++){
		Node nNode = s.item(i);
		if(nNode.getNodeType() == Node.ELEMENT_NODE){

			Element e = (Element) nNode;

			System.out.println(e.getTextContent());
			writer.println(e.getTextContent());

		}
	 }
        writer.close();
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
}
