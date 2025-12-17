//
//  ProjectTealApp.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import SwiftUI

@main
struct ProjectTealApp: App {
    private let environment = AppEnvironment.shared

    var body: some Scene {
        WindowGroup {
            AppRootView(environment: environment)
        }
    }
}
