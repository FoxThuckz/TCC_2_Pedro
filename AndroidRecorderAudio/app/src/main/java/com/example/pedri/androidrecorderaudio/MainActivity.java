package com.example.pedri.androidrecorderaudio;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import android.widget.Toast;
import android.widget.*;

import java.io.BufferedReader;
import java.io.File;

import com.amazonaws.auth.CognitoCachingCredentialsProvider;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferListener;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferObserver;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferState;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferUtility;
import com.amazonaws.regions.Region;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;

import org.w3c.dom.Text;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class MainActivity<var> extends AppCompatActivity {

    //Declarando as vaiavéis
    Button btnRecord, btnStopRecord, btnPlay, btnStop, btnEnviar,bntApp;
    String pathSave = "";
    MediaRecorder mediaRecorder;
    MediaPlayer mediaPlayer;
    TextView text;
    final String FILENAME = "freq_o.txt";
    final int REQUEST_PERMISSION_CODE = 1000;


    public void upload(View v){
        File sdcard = Environment.getExternalStorageDirectory();
        File file = new File(sdcard, "fox.3gp");
        CognitoCachingCredentialsProvider credentialsProvider = new CognitoCachingCredentialsProvider(
                getApplicationContext(),
                "us-east-2:0894d067-ddc1-42c8-98e8-59ac18f84a7c", // Identity pool ID
                Regions.US_EAST_2 // Region
        );
        AmazonS3 s3= new AmazonS3Client(credentialsProvider);
        s3.setRegion(Region.getRegion(Regions.SA_EAST_1));
        TransferUtility transferUtility = new TransferUtility(s3, this);
        TransferObserver transferObserver =
                transferUtility.upload("thuckz-android", "fox.3gp",file);
        transferObserver.setTransferListener(new TransferListener() {
            @Override
            public void onStateChanged(int id, TransferState state) {
                if(state == TransferState.COMPLETED) {
                    Toast.makeText(MainActivity.this, "Arquivo foi enviado com Sucesso!", Toast.LENGTH_SHORT).show();
                }
                else
                    Toast.makeText(MainActivity.this, " ", Toast.LENGTH_SHORT).show();
            }
            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
            }
            @Override
            public void onError(int id, Exception ex) {
                Log.e("fox.3gp","Erro" + ex.getMessage());

            }
        });
    }
    public void download(View v) throws IOException {
        File downloadFromS3 = new File("/sdcard/freq_o.txt");

        CognitoCachingCredentialsProvider credentialsProvider = new CognitoCachingCredentialsProvider(
                getApplicationContext(),
                "us-east-2:0894d067-ddc1-42c8-98e8-59ac18f84a7c", // Identity pool ID
                Regions.US_EAST_2 // Region
        );
        AmazonS3 s3= new AmazonS3Client(credentialsProvider);
        s3.setRegion(Region.getRegion(Regions.SA_EAST_1));
        TransferUtility transferUtility = new TransferUtility(s3, this);

        TransferObserver transferObserver = transferUtility.download("thuckz-android", "freq_o.txt", downloadFromS3);
        transferObserver.setTransferListener(new TransferListener() {
            @Override
            public void onStateChanged(int id, TransferState state) {
                if(state == TransferState.COMPLETED) {
                    Toast.makeText(MainActivity.this, "Download Completed", Toast.LENGTH_SHORT).show();
                }
                else
                    Toast.makeText(MainActivity.this, "Failed..", Toast.LENGTH_SHORT).show();
            }
            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
            }
            @Override
            public void onError(int id, Exception ex) {
                Log.e("freq_o.txt","Erro" + ex.getMessage());
            }
        });

    }
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!checkPermissionFromDevice())
            requestPermissions();

        text = findViewById(R.id.text);
        btnEnviar = (Button) findViewById(R.id.btnEnviar);
        btnPlay = (Button) findViewById(R.id.btnPlay);
        btnRecord = (Button) findViewById(R.id.btnStartRecord);
        btnStop = (Button) findViewById(R.id.btnStop);
        btnStopRecord = (Button) findViewById(R.id.btnStopRecord);
        bntApp = (Button) findViewById(R.id.bntApp);

        btnRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (checkPermissionFromDevice()) {

                    pathSave = Environment.getExternalStorageDirectory()
                            .getAbsolutePath() + "/fox.3gp";
                    setupMediaRecorder();
                    try {
                        mediaRecorder.prepare();
                        mediaRecorder.start();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    btnPlay.setEnabled(false);
                    btnStop.setEnabled(false);

                    Toast.makeText(MainActivity.this, "Recording...", Toast.LENGTH_SHORT).show();

                }
                else{
                    requestPermissions();
                }
            }

        });

        btnStopRecord.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mediaRecorder.stop();
                btnStopRecord.setEnabled(true);
                btnRecord.setEnabled(true);
                btnPlay.setEnabled(true);
                btnStop.setEnabled(false);

            }
        });
        btnPlay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                btnStop.setEnabled(true);
                btnStopRecord.setEnabled(false);
                btnRecord.setEnabled(false);

                mediaPlayer = new MediaPlayer();
                try {
                    mediaPlayer.setDataSource(pathSave);
                    mediaPlayer.prepare();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                mediaPlayer.start();
                Toast.makeText(MainActivity.this, "Playing...", Toast.LENGTH_SHORT).show();
            }
        });

        btnStop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                btnStopRecord.setEnabled(true);
                btnRecord.setEnabled(true);
                btnStop.setEnabled(false);
                btnPlay.setEnabled(true);

                if(mediaPlayer != null)
                {
                    mediaPlayer.stop();
                    mediaPlayer.release();
                    setupMediaRecorder();

                }
            }
        });
    }
    public void readFile(View v){
        if(isExternalStorageReadable()){
            StringBuilder sb = new StringBuilder();
            try{
                File textFile = new File(Environment.getExternalStorageDirectory(), FILENAME);
                FileInputStream fis = new FileInputStream(textFile);

                if(fis != null){
                    InputStreamReader isr = new InputStreamReader(fis);
                    BufferedReader buff = new BufferedReader(isr);

                    String line = null;
                    while((line = buff.readLine()) != null){
                        sb.append(line + "\n");
                    }
                    fis.close();
                }
                text.setText(sb);
            }catch (IOException e){
                e.printStackTrace();
            }
        }else{
            Toast.makeText(this, "Cannot Read from External Storage.", Toast.LENGTH_SHORT).show();
        }
    }
    private boolean isExternalStorageReadable() {
        if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())
                || Environment.MEDIA_MOUNTED_READ_ONLY.equals(Environment.getExternalStorageState())) {
            Log.i("State", "Yes, it is readable!");
            return true;
        } else {
            return false;
        }
    }
    private void setupMediaRecorder() {
        mediaRecorder = new MediaRecorder();
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
        mediaRecorder.setAudioEncoder(MediaRecorder.OutputFormat.AMR_NB);
        mediaRecorder.setOutputFile(pathSave);
    }
    private void requestPermissions() {
        ActivityCompat.requestPermissions(this,new String[]{
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.RECORD_AUDIO
        },REQUEST_PERMISSION_CODE);

    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode)
        {
            case REQUEST_PERMISSION_CODE:
            {
                if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                    Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
                else
                    Toast.makeText(this, "Permission Danied", Toast.LENGTH_SHORT).show();
            }
            break;
        }
    }
    private boolean checkPermissionFromDevice() {
        int read_external_storege_result = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
        int write_external_storege_result = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int record_audio_result = ContextCompat.checkSelfPermission(this,Manifest.permission.RECORD_AUDIO);
        return write_external_storege_result == PackageManager.PERMISSION_GRANTED &&
                record_audio_result == PackageManager.PERMISSION_GRANTED && read_external_storege_result == PackageManager.PERMISSION_GRANTED;
    }
}