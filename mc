Activity main xml

<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <EditText
        android:id="@+id/edit_name"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:ems="10"
        android:inputType="text"
        android:text=""
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.92"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.109" />

    <EditText
        android:id="@+id/edit_age"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:ems="10"
        android:inputType="text"
        android:text=""
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.92"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.256" />

    <TextView
        android:id="@+id/name"
        android:layout_width="128dp"
        android:layout_height="46dp"
        android:gravity="center"
        android:text="Name"
        android:textSize="20dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.106"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.109" />

    <TextView
        android:id="@+id/age"
        android:layout_width="128dp"
        android:layout_height="46dp"
        android:gravity="center"
        android:text="Age"
        android:textSize="20dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.106"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.256" />

    <Button
        android:id="@+id/insert"
        android:layout_width="120dp"
        android:layout_height="49dp"
        android:text="Insert"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.13"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.417" />

    <Button
        android:id="@+id/view"
        android:layout_width="120dp"
        android:layout_height="49dp"
        android:text="View"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.292"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.417" />

    <Button
        android:id="@+id/delete"
        android:layout_width="120dp"
        android:layout_height="49dp"
        android:text="Delete"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.542"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.417" />

    <TextView
        android:id="@+id/textView4"
        android:layout_width="338dp"
        android:layout_height="117dp"
        android:text=""
        android:textSize="20dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintHorizontal_bias="0.41"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintVertical_bias="0.628" />

    <Button
        android:id="@+id/update"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="284dp"
        android:layout_marginEnd="32dp"
        android:text="Update"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toTopOf="parent" />

</androidx.constraintlayout.widget.ConstraintLayout>



MainActivity Kotlin

package com.example.myapplication

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val name = findViewById<EditText>(R.id.edit_name)
        val age = findViewById<EditText>(R.id.edit_age)
        val insert = findViewById<Button>(R.id.insert)
        val view = findViewById<Button>(R.id.view)
        val delete = findViewById<Button>(R.id.delete)
        val update=findViewById<Button>(R.id.update)
        val result = findViewById<TextView>(R.id.textView4)

        val context = this
        val db = Database(context)

        insert.setOnClickListener {
            val user = Users(name.text.toString(), age.text.toString().toInt())
            db.insertData(user)
        }

        view.setOnClickListener {
            result.text = "" // Clear previous text
            val data = db.readData()
            for (i in data.indices) {
                result.append(data[i].name + " " + data[i].age + "\n")
            }
        }

        delete.setOnClickListener {
            val userName = name.text.toString()
            db.deleteData(userName)
            name.text.clear()
            age.text.clear()
            result.text = "User '$userName' deleted successfully."
        }

        update.setOnClickListener {
            val userName = name.text.toString()
            val userAge = age.text.toString().toInt()
            db.updateData(userName, userAge)
            result.text = "User '$userName' updated successfully."
        }
    }
}


Database Kotlin

package com.example.myapplication

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.widget.Toast

class Database(var context: Context): SQLiteOpenHelper(context, "UserDB", null,1) {
    override fun onCreate(db: SQLiteDatabase?) {
        val createTable = "CREATE TABLE Users(Name VARCHAR(50), Age INTEGER)";
        db?.execSQL(createTable)
    }

    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {
        TODO("Not yet implemented")
    }

    fun insertData(user: Users){
        val db = this.writableDatabase
        var cv = ContentValues()
        cv.put("Name", user.name)
        cv.put("Age", user.age)

        var result = db.insert("Users", null, cv)
        if (result == -1L) {
            Toast.makeText(context, "Failed", Toast.LENGTH_SHORT).show()
        }
        else
            Toast.makeText(context, "Success", Toast.LENGTH_SHORT).show()
    }

    fun readData(): MutableList<Users>{
        var list: MutableList<Users> = ArrayList()
        val db = this.readableDatabase
        val result = db.rawQuery("SELECT * FROM Users", null)
        if(result.moveToFirst()){
            do{
                var user = Users()
                user.name = result.getString(0)
                user.age = result.getString(1).toInt()
                list.add(user)
            }while (result.moveToNext())
        }
        return list
    }

    fun deleteData(name: String) {
        val db = this.writableDatabase
        val result = db.delete("Users", "Name = ?", arrayOf(name))
        if (result == 0) {
            Toast.makeText(context, "Failed to delete", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "Successfully deleted", Toast.LENGTH_SHORT).show()
        }
    }

    fun updateData(oldName: String, newAge: Int) {
        val db = this.writableDatabase
        val cv = ContentValues()
        cv.put("Age", newAge)
        val result = db.update("Users", cv, "Name = ?", arrayOf(oldName))
        if (result == -1) {
            Toast.makeText(context, "Failed to update", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(context, "Successfully updated", Toast.LENGTH_SHORT).show()
        }
    }

}


User Kotlin

package com.example.myapplication

class Users {
    var name : String = ""
    var age : Int = 0

    constructor(){}
    constructor(name: String, age:Int){
        this.name = name
        this.age = age
    }
}
