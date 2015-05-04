/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    PythonScriptExcecutorCustomizer.java
 *    Copyright (C) 2015 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.beans;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Constructor;
import java.util.Properties;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
import javax.swing.text.BadLocationException;
import javax.swing.text.DefaultStyledDocument;

import weka.core.Environment;
import weka.core.EnvironmentHandler;
import weka.core.Utils;
import weka.gui.PropertySheetPanel;
import weka.gui.visualize.VisualizeUtils;
import weka.python.PythonSession;

/**
 * Customizer for the PythonScriptExecutor
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public class PythonScriptExecutorCustomizer extends JPanel implements
  BeanCustomizer, EnvironmentHandler, CustomizerCloseRequester {

  /** editor setup */
  public final static String PROPERTIES_FILE =
    "weka/gui/scripting/Jython.props";

  protected ModifyListener m_modifyL = null;
  protected Environment m_env = Environment.getSystemWide();

  protected Window m_parent;

  protected PythonScriptExecutor m_executor;
  protected JTextPane m_scriptEditor;

  /** If loading a user script from a file at runtime - overrides in editor one */
  protected FileEnvironmentField m_scriptLoader;

  private boolean m_pyAvailable = true;

  /** Just used to get about info from the PythonScriptExecutor */
  protected PropertySheetPanel m_tempEditor = new PropertySheetPanel();

  /** Menu bar to use */
  protected JMenuBar m_menuBar;

  /** Text field for variables to extract from python */
  protected EnvironmentField m_outputVarsField = new EnvironmentField();

  /** Whether to output debugging info (both client and python server) */
  protected JCheckBox m_outputDebugInfo = new JCheckBox();

  public PythonScriptExecutorCustomizer() {
    setLayout(new BorderLayout());
  }

  private void setup() {
    Properties props = null;

    try {
      props = Utils.readProperties(PROPERTIES_FILE);
    } catch (Exception ex) {
      ex.printStackTrace();
      props = new Properties();
    }
    final Properties propsCopy = props;

    // check for SyntaxDocument
    boolean syntaxDocAvailable = true;
    try {
      Class.forName("weka.gui.scripting.SyntaxDocument");
    } catch (Exception ex) {
      syntaxDocAvailable = false;
    }

    m_scriptEditor = new JTextPane();

    if (props.getProperty("Syntax", "false").equals("true")
      && syntaxDocAvailable) {
      try {
        Class syntaxClass = Class.forName("weka.gui.scripting.SyntaxDocument");
        Constructor constructor = syntaxClass.getConstructor(Properties.class);
        Object doc = constructor.newInstance(props);
        m_scriptEditor.setDocument((DefaultStyledDocument) doc);
        m_scriptEditor.setBackground(VisualizeUtils.processColour(
          props.getProperty("BackgroundColor", "white"), Color.WHITE));
      } catch (Exception ex) {
        ex.printStackTrace();
      }
    } else {
      m_scriptEditor.setForeground(VisualizeUtils.processColour(
        props.getProperty("ForegroundColor", "black"), Color.BLACK));
      m_scriptEditor.setBackground(VisualizeUtils.processColour(
        props.getProperty("BackgroundColor", "white"), Color.WHITE));
      m_scriptEditor.setFont(new Font(props.getProperty("FontName",
        "monospaced"), Font.PLAIN, Integer.parseInt(props.getProperty(
        "FontSize", "12"))));
    }

    // check python availability

    m_pyAvailable = true;
    String envEvalResults = null;
    Exception envEvalEx = null;
    if (!PythonSession.pythonAvailable()) {
      // try initializing
      try {
        if (!PythonSession.initSession("python", m_executor.getDebug())) {
          envEvalResults = PythonSession.getPythonEnvCheckResults();
          m_pyAvailable = false;
        }
      } catch (Exception ex) {
        envEvalEx = ex;
        m_pyAvailable = false;
        ex.printStackTrace();
      }
    }

    try {
      if (m_pyAvailable) {
        m_scriptEditor.getDocument().insertString(0,
          m_executor.getPythonScript(), null);
      } else {
        String message =
          "Python does not seem to be available:\n\n" + envEvalResults != null ? envEvalResults
            : envEvalEx.getMessage();
      }
    } catch (BadLocationException ex) {
      ex.printStackTrace();
    }

    JPanel editorPan = new JPanel();
    editorPan.setLayout(new BorderLayout());

    JPanel topHolder = new JPanel(new BorderLayout());
    JScrollPane editorScroller = new JScrollPane(m_scriptEditor);
    editorScroller.setBorder(BorderFactory.createTitledBorder("Python Script"));
    topHolder.add(editorScroller, BorderLayout.NORTH);
    editorPan.add(topHolder, BorderLayout.NORTH);
    add(editorPan, BorderLayout.CENTER);
    Dimension d = new Dimension(450, 200);
    m_scriptEditor.setMinimumSize(d);
    m_scriptEditor.setPreferredSize(d);

    JPanel varsPanel = new JPanel(new GridLayout(0, 2));
    JLabel varsLab =
      new JLabel("Variables to get from python", SwingConstants.RIGHT);
    varsLab.setToolTipText("Output these variables from python");
    varsPanel.add(varsLab);
    varsPanel.add(m_outputVarsField);
    if (m_executor.getVariablesToGetFromPython() != null) {
      m_outputVarsField.setText(m_executor.getVariablesToGetFromPython());
    }
    topHolder.add(varsPanel, BorderLayout.SOUTH);

    JPanel scriptLoaderP = new JPanel();
    scriptLoaderP.setLayout(new GridLayout(0, 2));
    JLabel loadL = new JLabel("Load script file", SwingConstants.RIGHT);
    loadL
      .setToolTipText("Load script file at run-time - overides editor script");
    varsPanel.add(loadL);
    m_scriptLoader = new FileEnvironmentField(m_env);
    if (m_executor.getScriptFile() != null
      && m_executor.getScriptFile().length() > 0) {
      m_scriptLoader.setText(m_executor.getScriptFile());
    }
    varsPanel.add(m_scriptLoader);

    JLabel debugL = new JLabel("Output debugging info", SwingConstants.RIGHT);
    varsPanel.add(debugL);
    varsPanel.add(m_outputDebugInfo);
    m_outputDebugInfo.setSelected(m_executor.getDebug());

    JPanel aboutP = m_tempEditor.getAboutPanel();
    add(aboutP, BorderLayout.NORTH);

    addButtons();

    final JFileChooser fileChooser = new JFileChooser();
    fileChooser.setAcceptAllFileFilterUsed(true);
    fileChooser.setMultiSelectionEnabled(false);

    m_menuBar = new JMenuBar();

    JMenu fileM = new JMenu();
    m_menuBar.add(fileM);
    fileM.setText("File");
    fileM.setMnemonic('F');

    JMenuItem newItem = new JMenuItem();
    fileM.add(newItem);
    newItem.setText("New");
    newItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_N,
      KeyEvent.CTRL_MASK));

    final boolean syntaxAvail = syntaxDocAvailable;
    final Properties propsC = props;
    newItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (propsCopy.getProperty("Syntax", "false").equals("true")
          && syntaxAvail) {
          try {
            Class syntaxClass =
              Class.forName("weka.gui.scripting.SyntaxDocument");
            Constructor constructor =
              syntaxClass.getConstructor(Properties.class);
            Object doc = constructor.newInstance(propsC);
            m_scriptEditor.setDocument((DefaultStyledDocument) doc);
            m_scriptEditor.setBackground(VisualizeUtils.processColour(
              propsC.getProperty("BackgroundColor", "white"), Color.WHITE));
          } catch (Exception ex) {
            ex.printStackTrace();
          }
        } else {
          m_scriptEditor.setText("");
        }
      }
    });

    JMenuItem loadItem = new JMenuItem();
    fileM.add(loadItem);
    loadItem.setText("Open File...");
    loadItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O,
      KeyEvent.CTRL_MASK));
    loadItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        int retVal =
          fileChooser.showOpenDialog(PythonScriptExecutorCustomizer.this);
        if (retVal == JFileChooser.APPROVE_OPTION) {
          // boolean ok = m_script.open(fileChooser.getSelectedFile());
          StringBuffer sb = new StringBuffer();
          try {
            BufferedReader br =
              new BufferedReader(new FileReader(fileChooser.getSelectedFile()));
            String line = null;
            while ((line = br.readLine()) != null) {
              sb.append(line).append("\n");
            }
            m_scriptEditor.getDocument().insertString(0, sb.toString(), null);
            br.close();
          } catch (Exception ex) {
            JOptionPane.showMessageDialog(PythonScriptExecutorCustomizer.this,
              "Couldn't open file '" + fileChooser.getSelectedFile() + "'!");
            ex.printStackTrace();
          }
        }
      }
    });

    JMenuItem saveAsItem = new JMenuItem();
    fileM.add(saveAsItem);

    saveAsItem.setText("Save As...");
    saveAsItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_A,
      KeyEvent.CTRL_MASK));
    saveAsItem.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // save(fileChooser);
        if (m_scriptEditor.getText() != null
          && m_scriptEditor.getText().length() > 0) {

          int retVal =
            fileChooser.showSaveDialog(PythonScriptExecutorCustomizer.this);
          if (retVal == JFileChooser.APPROVE_OPTION) {
            try {
              BufferedWriter bw =
                new BufferedWriter(
                  new FileWriter(fileChooser.getSelectedFile()));
              bw.write(m_scriptEditor.getText());
              bw.flush();
              bw.close();
            } catch (Exception ex) {
              JOptionPane.showMessageDialog(
                PythonScriptExecutorCustomizer.this,
                "Unable to save script file '" + fileChooser.getSelectedFile()
                  + "'!");
              ex.printStackTrace();
            }
          }
        }
      }
    });
  }

  private void addButtons() {
    JButton okBut = new JButton("OK");
    JButton cancelBut = new JButton("Cancel");

    JPanel butHolder = new JPanel();
    butHolder.setLayout(new GridLayout(1, 2));
    butHolder.add(okBut);
    butHolder.add(cancelBut);
    add(butHolder, BorderLayout.SOUTH);

    okBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        closingOK();

        m_parent.dispose();
      }
    });

    cancelBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        closingCancel();

        m_parent.dispose();
      }
    });
  }

  public void closingOK() {
    if (!m_pyAvailable) {
      return;
    }

    if (!m_scriptEditor.getText().equals(m_executor.getPythonScript())) {
      if (m_modifyL != null) {
        m_modifyL.setModifiedStatus(PythonScriptExecutorCustomizer.this, true);
      }
    }

    if (!m_scriptLoader.getText().equals(m_executor.getScriptFile())) {
      if (m_modifyL != null) {
        m_modifyL.setModifiedStatus(PythonScriptExecutorCustomizer.this, true);
      }
    }

    m_executor.setPythonScript(m_scriptEditor.getText());
    m_executor.setScriptFile(m_scriptLoader.getText());
    m_executor.setVariablesToGetFromPython(m_outputVarsField.getText());
    m_executor.setDebug(m_outputDebugInfo.isSelected());
  }

  public void closingCancel() {
    // nothing to do
  }

  @Override
  public void setModifiedListener(ModifyListener l) {
    m_modifyL = l;
  }

  @Override
  public void setObject(Object bean) {
    if (bean instanceof PythonScriptExecutor) {
      m_executor = (PythonScriptExecutor) bean;
      m_tempEditor.setTarget(bean);
      setup();
    }
  }

  @Override
  public void setParentWindow(Window parent) {
    m_parent = parent;

    if (parent instanceof javax.swing.JDialog) {
      ((javax.swing.JDialog) m_parent).setJMenuBar(m_menuBar);
      ((javax.swing.JDialog) m_parent).setTitle("Python Script Editor");
    }
  }

  @Override
  public void setEnvironment(Environment env) {
    m_env = env;
  }
}